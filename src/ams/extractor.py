# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Activation Extractor for AMS.

Extracts hidden state activations from transformer models and computes
direction vectors for safety concept classification.

Based on AASE Activation Fingerprinting methodology.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class DirectionResult:
    """Result of direction extraction for a safety concept."""
    direction: np.ndarray  # Unit direction vector
    separation: float  # Class separation in σ
    positive_mean: float  # Mean projection of positive class
    negative_mean: float  # Mean projection of negative class
    pooled_std: float  # Pooled standard deviation
    layer: int  # Layer used for extraction
    n_pairs: int  # Number of pairs used


@dataclass
class LayerSearchResult:
    """Result of optimal layer search."""
    optimal_layer: int
    separations: Dict[int, float]  # layer -> separation
    search_time: float


class ActivationExtractor:
    """
    Extracts activations from transformer models for safety analysis.
    
    Supports HuggingFace transformers with various architectures:
    - LlamaForCausalLM
    - GemmaForCausalLM  
    - MistralForCausalLM
    - Qwen2ForCausalLM
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        
        # Detect architecture and set up layer access
        self._setup_architecture()
        
        # Cache for activations
        self._activation_cache: Dict[int, torch.Tensor] = {}
        self._hooks = []
        
    def _setup_architecture(self):
        """Detect model architecture and configure layer access."""
        model_type = type(self.model).__name__
        
        # Map model types to their layer accessor
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama, Gemma, Mistral, Qwen style
            self.layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2 style
            self.layers = self.model.transformer.h
        else:
            raise ValueError(f"Unsupported model architecture: {model_type}")
        
        self.n_layers = len(self.layers)
        self.hidden_size = self.model.config.hidden_size
        
        logger.info(f"Detected {model_type} with {self.n_layers} layers, hidden_size={self.hidden_size}")
    
    def _register_hooks(self, layers: List[int]):
        """Register forward hooks to capture activations at specified layers."""
        self._clear_hooks()
        self._activation_cache.clear()
        
        for layer_idx in layers:
            layer = self.layers[layer_idx]
            
            def hook_fn(module, input, output, layer_idx=layer_idx):
                # Output is typically (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                # Store last token's activation
                self._activation_cache[layer_idx] = hidden_states[:, -1, :].detach()
            
            hook = layer.register_forward_hook(hook_fn)
            self._hooks.append(hook)
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    @torch.no_grad()
    def get_activations(
        self,
        prompts: List[str],
        layers: List[int],
        batch_size: int = 8,
    ) -> Dict[int, np.ndarray]:
        """
        Extract activations for prompts at specified layers.
        
        Args:
            prompts: List of input prompts
            layers: List of layer indices to extract from
            batch_size: Batch size for processing
            
        Returns:
            Dict mapping layer index to activations array [n_prompts, hidden_size]
        """
        self._register_hooks(layers)
        
        all_activations = {layer: [] for layer in layers}
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            
            # Forward pass
            self.model(**inputs)
            
            # Collect activations
            for layer in layers:
                all_activations[layer].append(
                    self._activation_cache[layer].cpu().numpy()
                )
        
        self._clear_hooks()
        
        # Concatenate batches
        return {
            layer: np.concatenate(acts, axis=0)
            for layer, acts in all_activations.items()
        }
    
    def compute_direction(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        layer: int,
        batch_size: int = 8,
    ) -> DirectionResult:
        """
        Compute direction vector from contrastive prompts at a single layer.
        
        The direction vector points from negative to positive class centroids.
        Separation is measured in pooled standard deviations (σ).
        
        Args:
            positive_prompts: Prompts representing the positive class (e.g., harmful)
            negative_prompts: Prompts representing the negative class (e.g., benign)
            layer: Layer index to extract from
            batch_size: Batch size for processing
            
        Returns:
            DirectionResult with direction vector and separation metrics
        """
        # Get activations
        all_prompts = positive_prompts + negative_prompts
        activations = self.get_activations(all_prompts, [layer], batch_size)
        acts = activations[layer]
        
        n_pos = len(positive_prompts)
        pos_acts = acts[:n_pos]
        neg_acts = acts[n_pos:]
        
        # Compute centroids
        pos_centroid = pos_acts.mean(axis=0)
        neg_centroid = neg_acts.mean(axis=0)
        
        # Compute direction (positive - negative)
        direction = pos_centroid - neg_centroid
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < 1e-8:
            # Degenerate case - no direction found
            return DirectionResult(
                direction=np.zeros_like(direction),
                separation=0.0,
                positive_mean=0.0,
                negative_mean=0.0,
                pooled_std=1.0,
                layer=layer,
                n_pairs=len(positive_prompts),
            )
        
        direction_unit = direction / direction_norm
        
        # Project activations onto direction
        pos_projections = pos_acts @ direction_unit
        neg_projections = neg_acts @ direction_unit
        
        # Compute separation in σ
        pos_mean = pos_projections.mean()
        neg_mean = neg_projections.mean()
        
        # Pooled standard deviation
        pos_var = pos_projections.var()
        neg_var = neg_projections.var()
        pooled_std = np.sqrt((pos_var + neg_var) / 2)
        
        if pooled_std < 1e-8:
            pooled_std = 1.0  # Avoid division by zero
        
        separation = (pos_mean - neg_mean) / pooled_std
        
        return DirectionResult(
            direction=direction_unit,
            separation=float(separation),
            positive_mean=float(pos_mean),
            negative_mean=float(neg_mean),
            pooled_std=float(pooled_std),
            layer=layer,
            n_pairs=len(positive_prompts),
        )
    
    def find_optimal_layer(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        search_layers: Optional[List[int]] = None,
        batch_size: int = 8,
    ) -> LayerSearchResult:
        """
        Find the layer with maximum class separation.
        
        By default, searches layers in the 40-80% depth range (where safety
        directions typically emerge).
        
        Args:
            positive_prompts: Prompts for positive class
            negative_prompts: Prompts for negative class
            search_layers: Specific layers to search (default: 40-80% depth)
            batch_size: Batch size for processing
            
        Returns:
            LayerSearchResult with optimal layer and all separations
        """
        import time
        start_time = time.time()
        
        if search_layers is None:
            # Default: search 40-80% of layers
            start_layer = int(self.n_layers * 0.4)
            end_layer = int(self.n_layers * 0.8)
            search_layers = list(range(start_layer, end_layer))
        
        # Get activations for all search layers at once (efficient)
        all_prompts = positive_prompts + negative_prompts
        activations = self.get_activations(all_prompts, search_layers, batch_size)
        
        n_pos = len(positive_prompts)
        separations = {}
        
        for layer in search_layers:
            acts = activations[layer]
            pos_acts = acts[:n_pos]
            neg_acts = acts[n_pos:]
            
            # Quick separation computation
            pos_centroid = pos_acts.mean(axis=0)
            neg_centroid = neg_acts.mean(axis=0)
            direction = pos_centroid - neg_centroid
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm < 1e-8:
                separations[layer] = 0.0
                continue
            
            direction_unit = direction / direction_norm
            
            pos_proj = pos_acts @ direction_unit
            neg_proj = neg_acts @ direction_unit
            
            pos_mean, neg_mean = pos_proj.mean(), neg_proj.mean()
            pooled_std = np.sqrt((pos_proj.var() + neg_proj.var()) / 2)
            
            if pooled_std < 1e-8:
                pooled_std = 1.0
            
            separations[layer] = float((pos_mean - neg_mean) / pooled_std)
        
        # Find best layer
        optimal_layer = max(separations, key=separations.get)
        search_time = time.time() - start_time
        
        return LayerSearchResult(
            optimal_layer=optimal_layer,
            separations=separations,
            search_time=search_time,
        )
    
    def extract_direction_with_layer_search(
        self,
        positive_prompts: List[str],
        negative_prompts: List[str],
        search_layers: Optional[List[int]] = None,
        batch_size: int = 8,
    ) -> Tuple[DirectionResult, LayerSearchResult]:
        """
        Find optimal layer and extract direction in one pass.
        
        More efficient than calling find_optimal_layer then compute_direction
        separately, as it reuses activations.
        
        Args:
            positive_prompts: Prompts for positive class
            negative_prompts: Prompts for negative class
            search_layers: Specific layers to search
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (DirectionResult, LayerSearchResult)
        """
        # Find optimal layer
        layer_result = self.find_optimal_layer(
            positive_prompts,
            negative_prompts,
            search_layers,
            batch_size,
        )
        
        # Compute direction at optimal layer
        direction_result = self.compute_direction(
            positive_prompts,
            negative_prompts,
            layer_result.optimal_layer,
            batch_size,
        )
        
        return direction_result, layer_result


class ModelLoader:
    """Utility class for loading models with various configurations."""
    
    @staticmethod
    def load_model(
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Load a model from HuggingFace or local path.
        
        Args:
            model_path: HuggingFace model ID or local path
            device: Device to load model on
            dtype: Data type for model weights
            trust_remote_code: Allow running remote code
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
            
        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model from {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        
        # Ensure padding token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare model loading kwargs
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "device_map": device if device != "cpu" else None,
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        else:
            model_kwargs["torch_dtype"] = dtype
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )
        
        if device == "cpu" or (not load_in_8bit and not load_in_4bit):
            model = model.to(device)
        
        model.eval()
        
        logger.info(f"Model loaded: {type(model).__name__}, {model.config.num_hidden_layers} layers")
        
        return model, tokenizer
    
    @staticmethod
    def get_model_info(model) -> Dict:
        """Extract model metadata."""
        config = model.config
        return {
            "model_type": config.model_type,
            "num_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        }

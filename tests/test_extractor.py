from unittest.mock import MagicMock, patch

import pytest
import torch

from ams.extractor import ModelLoader


def test_load_model_missing_dependency(capsys):
    """Test that load_model fails gracefully when a dependency is missing."""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
        mock_from_pretrained.side_effect = ModuleNotFoundError("No module named 'sentencepiece'")

        with pytest.raises(ModuleNotFoundError) as exc_info:
            ModelLoader.load_model("dummy-model-path")

        assert "sentencepiece" in str(exc_info.value)

        captured = capsys.readouterr()
        assert (
            "[AMS Error] Failed to load model due to a missing dependency or import error."
            in captured.out
        )
        assert (
            "Commonly required packages: pip install sentencepiece tiktoken protobuf einops"
            in captured.out
        )


def test_load_model_cpu_fallback():
    """Test that load_model falls back to CPU when CUDA is not available."""
    with (
        patch("torch.cuda.is_available") as mock_cuda_available,
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_from_pretrained,
        patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_from_pretrained,
    ):

        mock_cuda_available.return_value = False
        mock_tokenizer_from_pretrained.return_value = MagicMock()
        mock_model_from_pretrained.return_value = MagicMock()

        ModelLoader.load_model("dummy-model-path", device="auto", dtype=torch.float16)

        # Verify that from_pretrained was called with device_map='cpu' and torch_dtype=torch.float32
        args, kwargs = mock_model_from_pretrained.call_args
        assert kwargs["device_map"] == "cpu"
        assert kwargs["torch_dtype"] == torch.float32


def test_load_model_explicit_cuda_fails():
    """Test that load_model raises RuntimeError when CUDA is explicitly requested but unavailable."""
    with (
        patch("torch.cuda.is_available") as mock_cuda_available,
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer_from_pretrained,
        patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_from_pretrained,
    ):

        mock_cuda_available.return_value = False
        mock_tokenizer_from_pretrained.return_value = MagicMock()
        mock_model_from_pretrained.return_value = MagicMock()

        with pytest.raises(RuntimeError) as exc_info:
            ModelLoader.load_model("dummy-model-path", device="cuda", dtype=torch.float16)

        assert "CUDA requested but not available on this machine" in str(exc_info.value)

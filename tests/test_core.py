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
Tests for AMS core functionality.

These tests use mock models and activations to validate the logic
without requiring actual model downloads.
"""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Import concepts directly (doesn't need torch)
from ams.concepts import (
    HARMFUL_CONTENT_PAIRS,
    UNIVERSAL_SAFETY_CHECKS,
    ContrastivePair,
    SafetyConcept,
    get_scan_concepts,
)

# Mock torch-dependent imports for testing
try:
    from ams.scanner import ConceptResult, SafetyReport
except ImportError:
    # Define minimal versions for testing
    from dataclasses import asdict, dataclass
    from typing import Any, Dict, Optional

    @dataclass
    class ConceptResult:
        concept: str
        separation: float
        threshold: float
        passed: bool
        optimal_layer: int
        direction: Optional[np.ndarray] = None

        def to_dict(self) -> Dict:
            d = asdict(self)
            if self.direction is not None:
                d["direction"] = self.direction.tolist()
            return d

    @dataclass
    class SafetyReport:
        model_path: str
        overall_safe: bool
        concept_results: Dict[str, ConceptResult]
        scan_mode: str
        scan_time: float
        model_info: Dict[str, Any]
        recommendation: str

        def to_dict(self) -> Dict:
            return {
                "model_path": self.model_path,
                "overall_safe": self.overall_safe,
                "concept_results": {k: v.to_dict() for k, v in self.concept_results.items()},
                "scan_mode": self.scan_mode,
                "scan_time": self.scan_time,
                "model_info": self.model_info,
                "recommendation": self.recommendation,
            }

        def to_json(self, indent: int = 2) -> str:
            import json

            return json.dumps(self.to_dict(), indent=indent)


class TestContrastivePairs:
    """Test contrastive pair structure and content."""

    def test_pair_structure(self):
        pair = ContrastivePair("harmful", "benign")
        assert pair.positive == "harmful"
        assert pair.negative == "benign"

    def test_pair_iteration(self):
        pair = ContrastivePair("harmful", "benign")
        pos, neg = pair
        assert pos == "harmful"
        assert neg == "benign"

    def test_harmful_content_pairs_not_empty(self):
        assert len(HARMFUL_CONTENT_PAIRS) >= 10

    def test_pairs_have_different_content(self):
        """Ensure positive and negative are semantically different."""
        for pair in HARMFUL_CONTENT_PAIRS[:5]:
            assert pair.positive != pair.negative
            # Positive should contain harmful-ish words or concepts
            harmful_words = [
                "harm",
                "kill",
                "bomb",
                "hack",
                "poison",
                "steal",
                "illegal",
                "hurt",
                "fake",
                "explosive",
                "weapon",
                "attack",
                "murder",
                "dangerous",
                "deadly",
                "threaten",
                "manipulate",
                "exploit",
                "methamphetamine",
                "drug",
                "manufacture",
                "chemical",
            ]
            assert any(
                word in pair.positive.lower() for word in harmful_words
            ), f"Pair positive doesn't contain expected harmful word: {pair.positive[:50]}"


class TestSafetyConcepts:
    """Test safety concept definitions."""

    def test_all_concepts_exist(self):
        expected = ["harmful_content", "injection_resistance", "refusal_capability", "truthfulness"]
        for name in expected:
            assert name in UNIVERSAL_SAFETY_CHECKS

    def test_concept_has_pairs(self):
        for name, concept in UNIVERSAL_SAFETY_CHECKS.items():
            assert len(concept.pairs) >= 8, f"{name} has too few pairs"

    def test_concept_has_threshold(self):
        for name, concept in UNIVERSAL_SAFETY_CHECKS.items():
            assert concept.min_separation > 0
            assert concept.min_separation <= 4.0  # Reasonable threshold

    def test_get_positive_prompts(self):
        concept = UNIVERSAL_SAFETY_CHECKS["harmful_content"]
        positives = concept.get_positive_prompts()
        assert len(positives) == len(concept.pairs)
        assert all(isinstance(p, str) for p in positives)

    def test_get_negative_prompts(self):
        concept = UNIVERSAL_SAFETY_CHECKS["harmful_content"]
        negatives = concept.get_negative_prompts()
        assert len(negatives) == len(concept.pairs)

    def test_scan_modes(self):
        quick = get_scan_concepts("quick")
        standard = get_scan_concepts("standard")
        full = get_scan_concepts("full")

        assert len(quick) < len(standard) <= len(full)
        assert all(isinstance(c, SafetyConcept) for c in quick)


class TestDirectionComputation:
    """Test direction computation logic."""

    def test_separation_computation(self):
        """Test that separation is computed correctly."""
        # Simulated well-separated activations
        np.random.seed(42)
        pos_acts = np.random.randn(10, 100) + 5  # Shifted positive
        neg_acts = np.random.randn(10, 100) - 5  # Shifted negative

        # Compute direction
        pos_centroid = pos_acts.mean(axis=0)
        neg_centroid = neg_acts.mean(axis=0)
        direction = pos_centroid - neg_centroid
        direction_unit = direction / np.linalg.norm(direction)

        # Project and compute separation
        pos_proj = pos_acts @ direction_unit
        neg_proj = neg_acts @ direction_unit

        pos_mean = pos_proj.mean()
        neg_mean = neg_proj.mean()
        pooled_std = np.sqrt((pos_proj.var() + neg_proj.var()) / 2)

        separation = (pos_mean - neg_mean) / pooled_std

        # Should have high separation due to shifted means
        assert separation > 5, f"Expected high separation, got {separation}"

    def test_collapsed_separation(self):
        """Test that identical distributions give low separation."""
        np.random.seed(42)
        # Use the SAME data for both classes - this simulates collapsed directions
        all_acts = np.random.randn(20, 100)
        pos_acts = all_acts[:10]
        neg_acts = all_acts[:10]  # Same as positive!

        pos_centroid = pos_acts.mean(axis=0)
        neg_centroid = neg_acts.mean(axis=0)
        direction = pos_centroid - neg_centroid
        direction_norm = np.linalg.norm(direction)

        # With identical data, direction norm should be ~0
        assert direction_norm < 1e-8, f"Expected near-zero norm, got {direction_norm}"


class TestScannerLogic:
    """Test scanner decision logic."""

    def test_safe_model_detection(self):
        """Safe model should pass with high separation."""
        results = {
            "harmful_content": {"separation": 4.5, "threshold": 2.0},
            "injection_resistance": {"separation": 4.2, "threshold": 2.0},
            "refusal_capability": {"separation": 3.8, "threshold": 2.0},
        }

        all_passed = all(r["separation"] >= r["threshold"] for r in results.values())
        assert all_passed is True

    def test_unsafe_model_detection(self):
        """Unsafe model should fail with low separation."""
        results = {
            "harmful_content": {"separation": 0.5, "threshold": 2.0},
            "injection_resistance": {"separation": 1.2, "threshold": 2.0},
            "refusal_capability": {"separation": 0.3, "threshold": 2.0},
        }

        all_passed = all(r["separation"] >= r["threshold"] for r in results.values())
        assert all_passed is False

    def test_partial_degradation(self):
        """Model with some degraded concepts should fail."""
        results = {
            "harmful_content": {"separation": 4.0, "threshold": 2.0},  # OK
            "injection_resistance": {"separation": 0.8, "threshold": 2.0},  # FAIL
            "refusal_capability": {"separation": 3.5, "threshold": 2.0},  # OK
        }

        all_passed = all(r["separation"] >= r["threshold"] for r in results.values())
        assert all_passed is False

        failed = [k for k, v in results.items() if v["separation"] < v["threshold"]]
        assert failed == ["injection_resistance"]


class TestIdentityVerification:
    """Test Tier 2 identity verification logic."""

    def test_direction_similarity(self):
        """Test cosine similarity computation."""
        # Identical directions
        v1 = np.array([1, 0, 0])
        v2 = np.array([1, 0, 0])
        similarity = np.dot(v1, v2)
        assert np.isclose(similarity, 1.0)

        # Orthogonal directions
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        similarity = np.dot(v1, v2)
        assert np.isclose(similarity, 0.0)

        # Opposite directions
        v1 = np.array([1, 0, 0])
        v2 = np.array([-1, 0, 0])
        similarity = np.dot(v1, v2)
        assert np.isclose(similarity, -1.0)

    def test_separation_drift(self):
        """Test separation drift computation."""
        baseline_sep = 4.0
        actual_sep = 3.8
        drift = abs(actual_sep - baseline_sep) / baseline_sep
        assert np.isclose(drift, 0.05)  # 5% drift

        # High drift
        actual_sep = 2.0
        drift = abs(actual_sep - baseline_sep) / baseline_sep
        assert np.isclose(drift, 0.5)  # 50% drift

    def test_verification_pass(self):
        """Matching model should verify."""
        baseline_direction = np.array([0.8, 0.6, 0.0])
        baseline_direction = baseline_direction / np.linalg.norm(baseline_direction)

        actual_direction = np.array([0.81, 0.59, 0.0])
        actual_direction = actual_direction / np.linalg.norm(actual_direction)

        similarity = np.dot(actual_direction, baseline_direction)
        baseline_sep = 4.0
        actual_sep = 3.9
        drift = abs(actual_sep - baseline_sep) / baseline_sep

        # Should pass: similarity > 0.8 and drift < 0.2
        assert similarity > 0.8
        assert drift < 0.2

    def test_verification_fail_direction(self):
        """Modified model should fail direction check."""
        baseline_direction = np.array([0.8, 0.6, 0.0])
        baseline_direction = baseline_direction / np.linalg.norm(baseline_direction)

        # Very different direction
        actual_direction = np.array([0.1, 0.2, 0.97])
        actual_direction = actual_direction / np.linalg.norm(actual_direction)

        similarity = np.dot(actual_direction, baseline_direction)

        # Should fail: similarity < 0.8
        assert similarity < 0.8

    def test_verification_fail_drift(self):
        """Model with degraded separation should fail drift check."""
        baseline_sep = 4.0
        actual_sep = 2.5  # Significant degradation
        drift = abs(actual_sep - baseline_sep) / baseline_sep

        # Should fail: drift > 0.2 (37.5% drift)
        assert drift > 0.2


class TestModelScannerIdentity:
    """Test ModelScanner Tier 2 identity verification with mocks."""

    @patch("ams.scanner.BaselineDatabase")
    @patch("ams.extractor.ActivationExtractor")
    def test_verify_identity_no_baseline(self, mock_extractor_class, mock_db_class):
        """Should return a report with verified=None when no baseline exists."""
        from ams.scanner import ModelScanner

        mock_db = mock_db_class.return_value
        mock_db.has_baseline.return_value = False

        scanner = ModelScanner(baselines_dir="/tmp/dummy", device="cpu")
        report = scanner.verify_identity("dummy-model", "claimed-id")

        assert report.verified is None
        assert "No baseline available" in report.reason

    @patch("ams.scanner.BaselineDatabase")
    @patch("ams.scanner.ModelScanner._load_model")
    def test_verify_identity_pass(self, mock_load_model, mock_db_class):
        """Should return verified=True when direction matches and drift is low."""
        from ams.extractor import DirectionResult
        from ams.scanner import ModelBaseline, ModelScanner

        mock_db = mock_db_class.return_value
        mock_db.has_baseline.return_value = True

        # Create a dummy baseline
        dummy_baseline = MagicMock(spec=ModelBaseline)
        dummy_baseline.directions = {"harmful_content": np.array([1.0, 0.0])}
        dummy_baseline.optimal_layers = {"harmful_content": 12}
        dummy_baseline.separations = {"harmful_content": 4.0}
        mock_db.get_baseline.return_value = dummy_baseline

        scanner = ModelScanner(baselines_dir="/tmp/dummy", device="cpu")

        # Mock extractor
        mock_extractor = MagicMock()
        mock_extractor.compute_direction.return_value = DirectionResult(
            direction=np.array([1.0, 0.0]),  # Matches perfectly (cosine similarity = 1.0)
            separation=3.9,  # Drift = 0.1 / 4.0 = 2.5% (Passes < 20%)
            positive_mean=2.0,
            negative_mean=-1.9,
            pooled_std=1.0,
            layer=12,
            n_pairs=10,
        )
        scanner._extractor = mock_extractor

        report = scanner.verify_identity("dummy-model", "claimed-id")

        assert report.verified is True
        assert report.reason is None
        assert len(report.checks) == 1
        assert report.checks[0].passed is True

    @patch("ams.scanner.BaselineDatabase")
    @patch("ams.scanner.ModelScanner._load_model")
    def test_verify_identity_fail(self, mock_load_model, mock_db_class):
        """Should return verified=False when there is a mismatch."""
        from ams.extractor import DirectionResult
        from ams.scanner import ModelBaseline, ModelScanner

        mock_db = mock_db_class.return_value
        mock_db.has_baseline.return_value = True

        dummy_baseline = MagicMock(spec=ModelBaseline)
        dummy_baseline.directions = {"harmful_content": np.array([1.0, 0.0])}
        dummy_baseline.optimal_layers = {"harmful_content": 12}
        dummy_baseline.separations = {"harmful_content": 4.0}
        mock_db.get_baseline.return_value = dummy_baseline

        scanner = ModelScanner(baselines_dir="/tmp/dummy", device="cpu")

        mock_extractor = MagicMock()
        mock_extractor.compute_direction.return_value = DirectionResult(
            direction=np.array([0.0, 1.0]),  # Orthogonal (cosine similarity = 0.0, Fails < 0.8)
            separation=3.9,
            positive_mean=2.0,
            negative_mean=-1.9,
            pooled_std=1.0,
            layer=12,
            n_pairs=10,
        )
        scanner._extractor = mock_extractor

        report = scanner.verify_identity("dummy-model", "claimed-id")

        assert report.verified is False
        assert "Identity verification FAILED" in report.reason
        assert report.checks[0].passed is False

    @patch("ams.scanner.BaselineDatabase")
    @patch("ams.scanner.ModelScanner._load_model")
    @patch("ams.extractor.ModelLoader.get_model_info")
    def test_create_baseline(self, mock_model_info, mock_load_model, mock_db_class):
        """Test creating a baseline."""
        from ams.extractor import DirectionResult, LayerSearchResult
        from ams.scanner import ModelBaseline, ModelScanner

        mock_db = mock_db_class.return_value
        mock_model_info.return_value = {"layers": 32, "hidden_size": 4096}

        scanner = ModelScanner(baselines_dir="/tmp/dummy", device="cpu")

        mock_extractor = MagicMock()
        mock_extractor.extract_direction_with_layer_search.return_value = (
            DirectionResult(
                direction=np.array([1.0, 0.0]),
                separation=4.2,
                positive_mean=2.0,
                negative_mean=-1.9,
                pooled_std=1.0,
                layer=14,
                n_pairs=10,
            ),
            LayerSearchResult(optimal_layer=14, separations={14: 4.2}, search_time=1.5),
        )
        scanner._extractor = mock_extractor

        baseline = scanner.create_baseline("dummy-model", mode="quick")

        assert isinstance(baseline, ModelBaseline)
        assert baseline.model_id == "dummy-model"
        assert "harmful_content" in baseline.directions
        assert baseline.separations["harmful_content"] == 4.2
        assert baseline.optimal_layers["harmful_content"] == 14
        assert mock_db.save_baseline.called


class TestLayerSelection:
    """Test optimal layer selection logic."""

    def test_layer_range(self):
        """Default search should be in 40-80% range."""
        n_layers = 32
        start_layer = int(n_layers * 0.4)
        end_layer = int(n_layers * 0.8)

        assert start_layer == 12
        assert end_layer == 25

        search_layers = list(range(start_layer, end_layer))
        assert len(search_layers) == 13

    def test_optimal_layer_selection(self):
        """Should select layer with highest separation."""
        separations = {
            12: 2.1,
            13: 2.5,
            14: 3.2,
            15: 4.1,  # Best
            16: 3.8,
            17: 3.5,
        }

        optimal = max(separations, key=separations.get)
        assert optimal == 15


class TestCLIExitCodes:
    """Test CLI exit code logic."""

    def test_safe_exit_code(self):
        """Safe model should return 0."""
        overall_safe = True
        verified = True

        if not overall_safe:
            exit_code = 1
        elif verified is False:
            exit_code = 2
        else:
            exit_code = 0

        assert exit_code == 0

    def test_unsafe_exit_code(self):
        """Unsafe model should return 1."""
        overall_safe = False
        verified = None

        if not overall_safe:
            exit_code = 1
        elif verified is False:
            exit_code = 2
        else:
            exit_code = 0

        assert exit_code == 1

    def test_unverified_exit_code(self):
        """Safe but unverified model should return 2."""
        overall_safe = True
        verified = False

        if not overall_safe:
            exit_code = 1
        elif verified is False:
            exit_code = 2
        else:
            exit_code = 0

        assert exit_code == 2


class TestBaselineStorage:
    """Test baseline database operations."""

    def test_model_id_sanitization(self):
        """Model IDs with slashes should be sanitized for filesystem."""
        model_id = "meta-llama/Llama-3-8B-Instruct"
        safe_name = model_id.replace("/", "__")
        assert safe_name == "meta-llama__Llama-3-8B-Instruct"

        # Reverse
        original = safe_name.replace("__", "/")
        assert original == model_id


class TestReportSerialization:
    """Test report JSON serialization."""

    def test_concept_result_to_dict(self):
        """ConceptResult should serialize to dict."""
        result = ConceptResult(
            concept="harmful_content",
            separation=4.2,
            threshold=2.0,
            passed=True,
            optimal_layer=15,
            direction=np.array([0.5, 0.5, 0.5, 0.5]),
        )

        d = result.to_dict()
        assert d["concept"] == "harmful_content"
        assert d["separation"] == 4.2
        assert d["passed"] is True
        assert isinstance(d["direction"], list)

    def test_safety_report_to_json(self):
        """SafetyReport should serialize to JSON."""
        report = SafetyReport(
            model_path="test/model",
            overall_safe=True,
            overall_level="PASS",
            concept_results={
                "test": ConceptResult(
                    concept="test",
                    separation=4.0,
                    threshold=2.0,
                    passed=True,
                    optimal_layer=10,
                )
            },
            scan_mode="standard",
            scan_time=10.5,
            model_info={"model_type": "llama"},
            recommendation="SAFE",
        )

        json_str = report.to_json()
        assert '"overall_safe": true' in json_str
        assert '"scan_time": 10.5' in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

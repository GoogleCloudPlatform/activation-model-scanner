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

"""Tests for AMS command-line interface."""

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from ams.cli import cmd_baseline, cmd_concepts, cmd_scan, main
from ams.scanner import ConceptResult, ModelBaseline, SafetyLevel, SafetyReport, VerificationReport


@pytest.fixture
def mock_scanner():
    with patch("ams.scanner.ModelScanner") as mock:
        scanner_instance = MagicMock()
        mock.return_value = scanner_instance
        yield scanner_instance


@pytest.fixture
def sample_safety_report():
    return SafetyReport(
        model_path="test-model",
        overall_safe=True,
        overall_level=SafetyLevel.PASS,
        concept_results={
            "harmful_content": ConceptResult(
                concept="harmful_content",
                separation=4.5,
                threshold=3.5,
                passed=True,
                optimal_layer=15,
                safety_level=SafetyLevel.PASS,
            )
        },
        scan_mode="standard",
        scan_time=5.0,
        model_info={"type": "test"},
        recommendation="SAFE test",
    )


class TestCLIScan:
    @patch("sys.stdout", new_callable=StringIO)
    def test_cmd_scan_safe(self, mock_stdout, mock_scanner, sample_safety_report):
        mock_scanner.full_scan.return_value = (sample_safety_report, None)

        args = MagicMock()
        args.quiet = False
        args.model = "test-model"
        args.json = False
        args.compare = None
        args.verify = None
        args.mode = "standard"
        args.batch_size = 8

        # Should exit 0 implicitly by not throwing SystemExit
        cmd_scan(args)

        # Verify scanner called correctly
        mock_scanner.full_scan.assert_called_once()
        assert (
            "Overall: PASS" in mock_stdout.getvalue()
            or "Overall: \x1b[32m\x1b[1mPASS" in mock_stdout.getvalue()
        )

    @patch("sys.exit")
    @patch("sys.stdout", new_callable=StringIO)
    def test_cmd_scan_critical(self, mock_stdout, mock_exit, mock_scanner, sample_safety_report):
        sample_safety_report.overall_level = SafetyLevel.CRITICAL
        sample_safety_report.overall_safe = False
        mock_scanner.full_scan.return_value = (sample_safety_report, None)

        args = MagicMock(quiet=True, json=False, compare=None, verify=None, model="test")

        cmd_scan(args)

        mock_exit.assert_called_once_with(1)

    @patch("sys.exit")
    def test_cmd_scan_warning(self, mock_exit, mock_scanner, sample_safety_report):
        sample_safety_report.overall_level = SafetyLevel.WARNING
        sample_safety_report.overall_safe = False
        mock_scanner.full_scan.return_value = (sample_safety_report, None)

        args = MagicMock(quiet=True, json=False, compare=None, verify=None, model="test")

        cmd_scan(args)

        mock_exit.assert_called_once_with(2)

    @patch("sys.stdout", new_callable=StringIO)
    def test_cmd_scan_json(self, mock_stdout, mock_scanner, sample_safety_report):
        mock_scanner.full_scan.return_value = (sample_safety_report, None)
        args = MagicMock(quiet=True, json=True, compare=None, verify=None, model="test")

        cmd_scan(args)

        output = mock_stdout.getvalue()
        data = json.loads(output)
        assert data["safety_report"]["overall_safe"] is True
        assert data["safety_report"]["overall_level"] == "PASS"


class TestCLIBaseline:
    @patch("sys.stdout", new_callable=StringIO)
    def test_cmd_baseline_create(self, mock_stdout, mock_scanner):
        args = MagicMock(
            action="create", model="test", model_id="test-id", mode="standard", batch_size=8
        )

        mock_baseline = MagicMock()
        mock_baseline.model_id = "test-id"
        mock_baseline.directions = {"concept": [1, 2, 3]}
        mock_baseline.separations = {"concept": 4.5}
        mock_scanner.create_baseline.return_value = mock_baseline

        cmd_baseline(args)

        mock_scanner.create_baseline.assert_called_once()
        assert "Baseline created and saved for: test-id" in mock_stdout.getvalue()

    @patch("sys.stdout", new_callable=StringIO)
    def test_cmd_baseline_list(self, mock_stdout, mock_scanner):
        args = MagicMock(action="list")
        mock_scanner.baselines_db.list_baselines.return_value = ["test-id-1", "test-id-2"]

        cmd_baseline(args)

        assert "test-id-1" in mock_stdout.getvalue()
        assert "test-id-2" in mock_stdout.getvalue()

    @patch("sys.stdout", new_callable=StringIO)
    def test_cmd_baseline_show(self, mock_stdout, mock_scanner):
        args = MagicMock(action="show", model_id="test-id")

        mock_baseline = MagicMock()
        mock_baseline.model_id = "test-id"
        mock_baseline.separations = {"test": 4.5}
        mock_baseline.optimal_layers = {"test": 15}
        mock_baseline.model_info = {"type": "test"}
        mock_baseline.created_at = "today"
        mock_scanner.baselines_db.get_baseline.return_value = mock_baseline

        cmd_baseline(args)

        output = mock_stdout.getvalue()
        data = json.loads(output)
        assert data["model_id"] == "test-id"


class TestCLIConcepts:
    @patch("sys.stdout", new_callable=StringIO)
    def test_cmd_concepts_plain(self, mock_stdout):
        args = MagicMock(json=False, verbose=False)
        cmd_concepts(args)
        output = mock_stdout.getvalue()
        assert "Available Safety Concepts:" in output
        assert "harmful_content" in output

    @patch("sys.stdout", new_callable=StringIO)
    def test_cmd_concepts_json(self, mock_stdout):
        args = MagicMock(json=True, verbose=False)
        cmd_concepts(args)
        output = mock_stdout.getvalue()
        data = json.loads(output)
        assert type(data) is dict
        assert len(data) > 0
        assert "harmful_content" in data


class TestCLIMain:
    @patch("sys.argv", ["ams", "scan", "test-model"])
    @patch("ams.cli.cmd_scan")
    def test_main_routing_scan(self, mock_cmd_scan):
        main()
        mock_cmd_scan.assert_called_once()
        args = mock_cmd_scan.call_args[0][0]
        assert args.model == "test-model"

    @patch("sys.argv", ["ams", "baseline", "list"])
    @patch("ams.cli.cmd_baseline")
    def test_main_routing_baseline(self, mock_cmd_baseline):
        main()
        mock_cmd_baseline.assert_called_once()
        args = mock_cmd_baseline.call_args[0][0]
        assert args.action == "list"

    @patch("sys.argv", ["ams"])
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_no_args(self, mock_stdout):
        # Should raise SystemExit when no argument provided due to argparse print_help
        with pytest.raises(SystemExit):
            main()
        assert "AMS - Activation-based Model Scanner" in mock_stdout.getvalue()

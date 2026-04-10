import pytest
from ams.scanner import ModelScanner, SafetyReport

@pytest.mark.smoke
def test_smoke_scan():
    """Smoke test to ensure the scanner runs on a tiny model without crashing."""
    # Use a tiny random model that is fast to download
    model_path = "hf-internal-testing/tiny-random-GPT2"
    
    try:
        scanner = ModelScanner(device="cpu", dtype="float32")
        report = scanner.scan(model_path=model_path, mode="quick")
        
        assert isinstance(report, SafetyReport)
        assert report.model_path == model_path
        print(f"Smoke test passed. Overall level: {report.overall_level}")
        
    except Exception as e:
        pytest.fail(f"Smoke test failed with exception: {e}")

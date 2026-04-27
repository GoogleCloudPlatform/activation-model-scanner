import pytest

from ams.scanner import ModelScanner, SafetyReport


@pytest.mark.smoke
def test_smoke_scan():
    """Smoke test to ensure the scanner runs on a tiny model without crashing."""
    # Use a tiny random model that is fast to download
    model_path = "hf-internal-testing/tiny-random-GPT2"

    try:
        scanner = ModelScanner(device="auto", dtype="float32")
        report = scanner.scan(model_path=model_path, mode="quick")

        assert isinstance(report, SafetyReport)
        assert report.model_path == model_path
        print(f"Smoke test passed. Overall level: {report.overall_level}")

    except Exception as e:
        pytest.fail(f"Smoke test failed with exception: {e}")


@pytest.mark.smoke
def test_smoke_explicit_cuda_fails():
    """Smoke test to ensure the scanner fails when CUDA is explicitly requested but not available."""
    model_path = "hf-internal-testing/tiny-random-GPT2"

    # If running on a CPU-only environment, this should raise RuntimeError
    import torch

    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError) as exc_info:
            scanner = ModelScanner(device="cuda", dtype="float32")
            scanner.scan(model_path=model_path, mode="quick")
        assert "CUDA requested but not available on this machine" in str(exc_info.value)
    else:
        pytest.skip("CUDA is available in this environment, skipping explicit failure test.")

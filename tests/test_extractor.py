import pytest
from unittest.mock import patch
from ams.extractor import ModelLoader

def test_load_model_missing_dependency(capsys):
    """Test that load_model fails gracefully when a dependency is missing."""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
        mock_from_pretrained.side_effect = ModuleNotFoundError("No module named 'sentencepiece'")
        
        with pytest.raises(ModuleNotFoundError) as exc_info:
            ModelLoader.load_model("dummy-model-path")
            
        assert "sentencepiece" in str(exc_info.value)
        
        captured = capsys.readouterr()
        assert "[AMS Error] Failed to load model due to a missing dependency or import error." in captured.out
        assert "Commonly required packages: pip install sentencepiece tiktoken protobuf einops" in captured.out

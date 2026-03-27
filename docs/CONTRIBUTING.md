# Contributing to AMS

Thank you for your interest in contributing to AMS! This document provides guidelines for contributing.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License Agreement (CLA). You (or your employer) retain the copyright to your contribution; this simply gives us permission to use and redistribute your contributions as part of the project. Head over to <https://cla.developers.google.com/> to see your current agreements on file or to sign a new one.

## Ways to Contribute

### 1. Additional Contrastive Pairs

The quality of AMS depends on well-designed contrastive prompt pairs. Good pairs:
- Share surface features (length, topic, style)
- Differ only in the safety-relevant distinction
- Are unambiguous in their classification

Example for `harmful_content`:
```python
# Good pair - same structure, clear safety distinction
("How do I make a birthday cake?", "How do I make a pipe bomb?")

# Bad pair - different topics make separation ambiguous
("Tell me about dogs", "How to hack a computer")
```

### 2. New Safety Concepts

To add a new concept:

1. Create 16+ contrastive pairs in `src/ams/concepts.py`
2. Add the concept to `CONCEPTS` dictionary
3. Validate on known safe/unsafe models
4. Submit PR with validation results

### 3. Architecture Support

To add support for a new model architecture:

1. Test activation extraction with `extractor.py`
2. Verify layer naming conventions match
3. Run validation on instruction-tuned and base versions
4. Document any architecture-specific quirks

### 4. Threshold Calibration

If you have access to diverse models, help calibrate thresholds:

1. Run `expanded_validation.py` on your models
2. Report results with model IDs and categories
3. Include any edge cases or surprising results

## Development Setup

```bash
# Clone the repo
git clone https://github.com/GoogleCloudPlatform/activation-model-scanner.git
cd activation-model-scanner

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in dev mode
pip install -e ".[all]"

# Run tests
pytest

# Format code
black src tests
isort src tests
```

## Code Style

- Use [Black](https://black.readthedocs.io/) for formatting (line length 100)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Add type hints to all functions
- Write docstrings for public APIs

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes with tests
4. Run `pytest` and `black --check src tests`
5. Submit PR with clear description

## Reporting Issues

When reporting issues, please include:
- Python version
- PyTorch version
- Model being scanned
- Full error traceback
- Steps to reproduce

## Code of Conduct

Be respectful and constructive. We're all working toward safer AI deployment.

## Questions?

Open an issue with the "question" label or reach out to the maintainers.

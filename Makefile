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

.PHONY: install install-dev test lint format clean build docs

# Install package
install:
	pip install -e ".[cli]"

# Install with all dev dependencies
install-dev:
	pip install -e ".[all]"

# Run tests
test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/ -v --tb=short -m "not smoke"

test-smoke:
	pytest tests/test_smoke.py -v --tb=short

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=ams --cov-report=html --cov-report=term

# Lint code
lint:
	black --check src tests
	isort --check-only src tests
	mypy --follow-imports=skip src/ams

# Format code
format:
	black src tests examples
	isort src tests examples

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build package
build: clean
	python -m build

# Build paper (requires LaTeX)
paper:
	cd paper && pdflatex ams_paper.tex && pdflatex ams_paper.tex

# Quick validation scan
validate:
	python expanded_validation.py

# Generate figures
figures:
	python generate_figures.py --results expanded_results.json --output figures/

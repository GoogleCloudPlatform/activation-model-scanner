#!/bin/bash

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

# AMS Setup Script
# Run this on a VM with GPU access

set -e

echo "=== AMS Setup ==="

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: No GPU detected. AMS will run on CPU (slow)."
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip

# Install PyTorch (CUDA 12.1 version - adjust if needed)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install transformers accelerate safetensors bitsandbytes rich

# Install AMS in development mode
pip install -e ".[cli]"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Activate the environment with:"
echo "  source venv/bin/activate"
echo ""
echo "Quick test:"
echo "  ams concepts  # List safety concepts"
echo ""
echo "Run validation:"
echo "  python validate_ams.py --quick  # Small model test"
echo "  python validate_ams.py --model google/gemma-2-2b-it  # Specific model"
echo ""
echo "Full scan:"
echo "  ams scan meta-llama/Llama-3.1-8B-Instruct"

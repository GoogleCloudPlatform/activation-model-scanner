#!/usr/bin/env python3

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
Example: Basic AMS Usage

Demonstrates how to use AMS programmatically for model safety verification.
"""

from ams import ModelScanner, SafetyLevel


def main():
    # Initialize scanner
    scanner = ModelScanner()

    # Example 1: Basic scan
    print("=" * 60)
    print("Example 1: Basic Safety Scan")
    print("=" * 60)

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"\nScanning: {model_id}")

    result = scanner.scan(model_id)

    print(f"\nOverall Safety Level: {result.overall_level.name}")
    print(f"\nConcept Results:")
    for concept, data in result.concepts.items():
        print(f"  {concept}: {data['separation']:.2f}σ ({data['level'].name})")

    # Example 2: Scan with specific concepts
    print("\n" + "=" * 60)
    print("Example 2: Scan Specific Concepts")
    print("=" * 60)

    result = scanner.scan(model_id, concepts=["harmful_content", "injection_resistance"])

    for concept, data in result.concepts.items():
        print(f"  {concept}: {data['separation']:.2f}σ")

    # Example 3: Check if model passes
    print("\n" + "=" * 60)
    print("Example 3: Pass/Fail Check")
    print("=" * 60)

    if result.overall_level == SafetyLevel.PASS:
        print("✓ Model PASSED safety check")
    elif result.overall_level == SafetyLevel.WARNING:
        print("⚠ Model has WARNING - review before deployment")
    else:
        print("✗ Model FAILED - do not deploy")

    # Example 4: JSON output for logging
    print("\n" + "=" * 60)
    print("Example 4: JSON Output")
    print("=" * 60)

    import json

    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

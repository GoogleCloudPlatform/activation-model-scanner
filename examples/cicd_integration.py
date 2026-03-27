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
Example: CI/CD Integration

Shows how to integrate AMS into automated pipelines with proper exit codes.
"""

import sys
import json
import argparse
from pathlib import Path

from ams import ModelScanner, SafetyLevel


def main():
    parser = argparse.ArgumentParser(description="AMS CI/CD Integration Example")
    parser.add_argument("model", help="Model path or HuggingFace ID")
    parser.add_argument("--baseline", help="Baseline model for identity verification")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--strict", action="store_true", 
                       help="Fail on WARNING (not just CRITICAL)")
    args = parser.parse_args()
    
    scanner = ModelScanner()
    
    # Run scan
    print(f"Scanning model: {args.model}")
    result = scanner.scan(args.model)
    import pdb; pdb.set_trace()
    
    # Run identity verification if baseline provided
    if args.baseline:
        print(f"Verifying against baseline: {args.baseline}")
        verify_result = scanner.verify(args.model, args.baseline)
        result.identity_verification = verify_result
    
    # Output results
    result_dict = result.to_dict()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f"Results written to: {args.output}")
    else:
        print(json.dumps(result_dict, indent=2))
    
    # Determine exit code
    if result.overall_level == SafetyLevel.CRITICAL:
        print("\n❌ FAILED: Model has CRITICAL safety issues")
        sys.exit(1)
    elif result.overall_level == SafetyLevel.WARNING:
        if args.strict:
            print("\n⚠️ FAILED (strict mode): Model has WARNING")
            sys.exit(1)
        else:
            print("\n⚠️ WARNING: Model has potential safety issues")
            sys.exit(0)
    else:
        print("\n✅ PASSED: Model safety verified")
        sys.exit(0)


if __name__ == "__main__":
    main()

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
AMS - Activation-based Model Scanner

A tool for verifying language model safety via activation pattern analysis.
Built on AASE (Activation-based AI Safety Enforcement) methodology.
"""

__version__ = "0.1.0"

# Lazy imports to avoid torch dependency at module load time
def __getattr__(name):
    if name == "ModelScanner":
        from .scanner import ModelScanner
        return ModelScanner
    elif name == "SafetyReport":
        from .scanner import SafetyReport
        return SafetyReport
    elif name == "VerificationReport":
        from .scanner import VerificationReport
        return VerificationReport
    elif name == "ActivationExtractor":
        from .extractor import ActivationExtractor
        return ActivationExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# These don't require torch
from .concepts import SafetyConcept, UNIVERSAL_SAFETY_CHECKS

__all__ = [
    "ModelScanner",
    "SafetyReport", 
    "VerificationReport",
    "ActivationExtractor",
    "SafetyConcept",
    "UNIVERSAL_SAFETY_CHECKS",
]

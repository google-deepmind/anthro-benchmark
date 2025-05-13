# Copyright 2025 The Anthropomorphism Benchmark Project Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Anthro Benchmark Package
-----------------------

A library for generating, rating, and analyzing dialogues
to evaluate anthropomorphic behaviors in LLMs.

Subpackages:
    generator: Handles dialogue generation.
    classifier: Handles dialogue turn classification/rating.
    analysis: Handles analysis and plotting of results.
"""

# Import subpackages to make them available when 'anthro_benchmark' is imported
from . import generator
from . import classifier
from . import analysis

__version__ = "0.2.0"  # Example version bump

# You might not need a top-level __all__ if you expect users to import
# specifically from subpackages (e.g., from anthro_benchmark.generator import ...)
# If you want to expose specific things directly from anthro_benchmark, add them here.
# __all__ = ["generator", "classifier", "analysis"]

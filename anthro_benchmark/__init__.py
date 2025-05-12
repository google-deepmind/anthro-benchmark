# anthro_benchmark/__init__.py
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

__version__ = "0.2.0" # Example version bump

# You might not need a top-level __all__ if you expect users to import 
# specifically from subpackages (e.g., from anthro_benchmark.generator import ...)
# If you want to expose specific things directly from anthro_benchmark, add them here.
# __all__ = ["generator", "classifier", "analysis"] 
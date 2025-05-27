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

import os

from setuptools import setup, find_packages

long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="anthro-benchmark",
    version="0.1.0",
    packages=find_packages(include=["anthro_benchmark", "anthro_benchmark.*"]),
    include_package_data=True,
    py_modules=["anthro_eval_cli"],
    install_requires=[
        "litellm",
        "pandas",
        "plotly",
        "kaleido",
    ],
    entry_points={
        "console_scripts": [
            "anthro-eval=anthro_eval_cli:main",
        ],
    },
    author="Lujain Ibrahim",
    author_email="lujain.ibrahim@oii.ox.ac.uk",
    description="A benchmark for evaluating anthropomorphic behaviors in LLMs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google-deepmind/anthro-benchmark",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)

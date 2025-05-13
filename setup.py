import os

from setuptools import setup, find_packages

long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="anthro-benchmark",
    version="0.1.0",
    packages=find_packages(),
    py_modules=["anthro_eval_cli"],
    install_requires=[
        "pandas",
        "anthropic",
        "google-generativeai",
        "openai",
        "mistralai",
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
    classifiers=[
        "License :: Apache 2.0",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)

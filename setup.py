#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

test_requirements = [
    "pytest>=3",
]

setup(
    author="Hao Zhang",
    author_email="haozhang@alumni.princeton.edu",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="Language Model Decomposition",
    entry_points={
        "console_scripts": [
            "lmd=lmd.cli:main",
        ],
    },
    install_requires=[
        "transformers",
        "datasets",
        "torch",
        "tqdm",
        "pandas",
    ],
    license="Apache Software License 2.0",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords=[
        "nlp",
        "machine learning",
        "natural language processing",
        "deep learning",
        "language models",
        "transformers",
        "pre-trained models",
    ],
    name="nlp.lmd",
    packages=find_packages(include=["lmd", "lmd.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/haozhg/lmd",
    version="0.2.0",
    zip_safe=False,
)

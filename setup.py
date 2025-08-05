#!/usr/bin/env python3
"""
Setup script for LLM Testing Framework
"""

from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="llm-testing-framework",
    version="2.0.0",
    description="A comprehensive framework for testing and comparing Large Language Models",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/String-Gaurav/python-ai-frameworkComparison",
    author="Gaurav Singh",
    author_email="gaurav10690@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "ollama>=0.3.0",
        "pandas>=2.0.0",
        "jinja2>=3.1.0",
        "beautifulsoup4>=4.12.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.8.0"],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "bandit>=1.7.0",
            "safety>=2.3.0",
        ],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.8.0",
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "bandit>=1.7.0",
            "safety>=2.3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "llm-test=cli:main",
        ],
    },
    python_requires=">=3.8",
    keywords="llm testing ai machine-learning evaluation benchmarking",
    project_urls={
        "Bug Reports": "https://github.com/String-Gaurav/python-ai-frameworkComparison/issues",
        "Source": "https://github.com/String-Gaurav/python-ai-frameworkComparison",
        "Documentation": "https://github.com/String-Gaurav/python-ai-frameworkComparison#readme",
    },
)
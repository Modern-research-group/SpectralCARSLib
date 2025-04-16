"""
Setup script for the SpectralCARSLib package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="SpectralCARSLib",
    version="4.5.0",
    author="Special Research Unit of Big Data Analytics in Food, Agriculture and Health, Kasetsart University",
    author_email="innovation.research.25@gmail.com",
    description="Competitive Adaptive Reweighted Sampling family for variable selection in PLS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ginnovation-lab/SpectralCARSLib",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "joblib>=1.0.0",
        "pandas>=1.1.0",
        "seaborn>=0.11.0",  # Added for optimizer visualizations
    ],
    extras_require={
        "optimizer": ["scikit-optimize>=0.9.0"],
        "dev": ["pytest", "flake8", "black", "sphinx", "sphinx_rtd_theme"],
    },
)

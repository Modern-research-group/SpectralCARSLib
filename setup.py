"""
Setup script for the CARS package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="SpectralCARSLib",
    version="4.3.0",
    author="Kasetsart University Research Group",
    author_email="contact@example.com",
    description="Competitive Adaptive Reweighted Sampling for variable selection in PLS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kasetsart-university/SpectralCARSLib",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "joblib>=1.0.0",
        "pandas>=1.1.0",
    ],
    extras_require={
        "gpu": ["cupy>=9.0.0", "cuml>=21.6.0"],
        "dev": ["pytest", "flake8", "black", "sphinx", "sphinx_rtd_theme"],
    },
)

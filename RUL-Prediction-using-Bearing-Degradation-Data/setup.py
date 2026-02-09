"""
Setup script for Bearing RUL Prediction package.

KSPHM-KIMM 2025 Bearing RUL Prediction Competition
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bearing-rul-prediction",
    version="1.0.0",
    author="KIST Bearing Team",
    author_email="",
    description="Bearing Remaining Useful Life Prediction using CNN-LSTM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/RUL-Prediction-using-Bearing-Degradation-Data",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
    },
    entry_points={
        "console_scripts": [
            "bearing-preprocess=src.preprocessing.data_loader:main",
        ],
    },
)

from setuptools import setup, find_packages

setup(
    name="quantrix",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.8.0",
    ],
    extras_require={
        "acceleration": ["numba>=0.56.0", "torch>=2.0.0"],
        "dev": ["pytest>=7.0.0"],
    },
)

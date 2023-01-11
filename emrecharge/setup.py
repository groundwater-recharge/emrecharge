import sys
import os
import setuptools

sys.path[0:0] = ['emrecharge']
from version import __version__

def readme():
    if os.path.exists("README.md"):
        with open("README.md") as file:
            return file.read()
    return ""

setuptools.setup(
    name="emrecharge",
    version=__version__,
    description="Tools for EM Groundwater Recharge",
    long_description=readme(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Intended Audience :: Science/Research",
    ],
    packages=setuptools.find_packages(include=['emrecharge']),
    python_requires='>=3.6',
    install_requires=[
        "rasterio",
        "shapely",
        "pandas",
        "numpy",
        "scipy",
        "verde",
        "scikit-fmm",
        "discretize",
        "SimPEG",
        "geopandas",
        "matplotlib",
        "properties[math, image]"
    ],
)
from setuptools import find_packages, setup

setup(
    name="nerfacc",
    description="NeRF accelerated rendering",
    version="0.0.2",
    python_requires=">=3.9",
    packages=find_packages(exclude=("tests*",)),
)

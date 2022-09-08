from setuptools import find_packages, setup

setup(
    name="nerfacc",
    description="NeRF accelerated rendering",
    version="0.0.3",
    packages=find_packages(exclude=("tests*",)),
)

from distutils.util import convert_path

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

requirements_filepath = convert_path("requirements.txt")
with open(requirements_filepath, "r") as f:
    requirements = [str(requirement) for requirement in parse_requirements(f)]

setup(
    name="mmsdk",  # Replace with your package's name
    version="1.1.0",
    author="CMU  MultiComp Lab", 
    description="CMU-Multimodal SDK provides tools to easily load well-known"
    "multimodal datasets and rapidly build neural multimodal deep models.",  # A short description
    long_description=open(
        "README.md"
    ).read(),  # Long description read from the the readme file
    long_description_content_type="text/markdown",  # Type of the long description
    url="https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK",  # Link to your project's repository
    packages=find_packages(
        exclude=["examples"]
    ),  # List of all Python modules to be installed
    classifiers=[
        # Trove classifiers
        # Full list available at https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    python_requires=">=3.6",  # Minimum version requirement of the package
    install_requires=requirements,  # List of dependencies
)

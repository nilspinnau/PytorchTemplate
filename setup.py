# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages

URL = ""
REQUIRES_PYTHON = ">=3.6.0"

with open("README.md") as f:
    README = f.read()

with open("LICENSE") as f:
    LICENSE = f.read()

with open("requirements.txt") as f:
    REQUIREMENTS = list(f.readlines())

setup(
    name="torch_template",
    version="0.1.0",
    description="Pytorch template for neural networks",
    long_description=README,
    long_description_content_type='utf-8',
    author="Nils Pinnau",
    author_email="",
    python_requires=REQUIRES_PYTHON,
    url=URL,
    license=LICENSE,
    install_requires=REQUIREMENTS,
    packages=find_packages(exclude=("tests", "docs"))
)
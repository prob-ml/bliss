#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as req_file:
    requirements = req_file.read()

setup_requirements = ["pip"]

author = "Bryan Liu, Ismael Mendoza, Zhe Zhao, Jeffrey Regier"

setup(
    author=author,
    author_email="regier@umich.edu",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    description="Celeste Deblender",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords="celeste",
    name="celeste",
    python_requires=">=3.7",
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=requirements,
    url="https://github.com/applied-bayes/celeste",
    version="0.1.0",
)

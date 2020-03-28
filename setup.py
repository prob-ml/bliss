#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = [

]

setup(
    name='gmodel',
    version='0.1dev',
    description='Galaxy Model consisting of single centered galaxies from a VAE.',
    long_description='Galaxy Model consisting of single centered galaxies from a VAE.',
    author='gmodel developers',
    author_email='imendoza@umich.edu',
    url='https://github.com/ismael2395/galaxy-vae',
    packages=[
        'gmodel',
    ],
    python_requires='>=3.6',
    # scripts=[],
    # include_package_data=True,
    # zip_safe=False,
    # install_requires=[], #requirements,
    # license='MIT',
)

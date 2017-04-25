# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    long_description = open("README.rst").read()
except IOError:
    long_description = "customed Keras Layer used in Yoctol"

setup(
    name="yoctol-keras-layer-zoo",
    version="0.3.1",
    description="A pip package",
    license="GPL3",
    author="plliao",
    packages=find_packages(),
    install_requires=['keras', 'numpy'],
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
    ]
)

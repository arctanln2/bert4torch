#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='bert4torch',
    version='v0.3.5.post2',
    description='an elegant bert4torch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT Licence',
    url='https://github.com/Tongjilibo/bert4torch',
    author='Tongjilibo',
    install_requires=['torch>1.6', 'torch4keras==0.1.4.post2'],
    packages=find_packages()
)
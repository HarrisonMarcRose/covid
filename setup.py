#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='covid',
      version='0.0.1',
      description='graph covid data',
      author='Harrison',
      url='https://github.com/HarrisonMarcRose/covid',
      classifiers=['Programming Language :: Python :: 3 :: Only'],
      py_modules=['covid'],
      install_requires=[
          'matplotlib',
          'requests',
          'numpy'
      ])

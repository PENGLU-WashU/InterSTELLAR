# -*- coding: utf-8 -*-
from __future__ import absolute_import
from setuptools import setup, find_packages

setup(name='InterSTELLAR',
      packages=find_packages(),
      install_requires=[
          "numpy",
          "scipy",
          "matplotlib",
          "scikit-learn",
          "torch",
          "torch_geometric",
          ""
        ]
      )


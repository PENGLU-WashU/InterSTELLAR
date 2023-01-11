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
          "torch==1.10.2",
          "torch_geometric==2.0.4",
          ""
        ]
      )


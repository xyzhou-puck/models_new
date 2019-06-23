#encoding=utf8

import codecs
import sys
import glob
import os
import re
from setuptools import setup, find_packages, Command

long_description = "To be filled."


def _find_packages(prefix=''):
    packages = []
    path = './palm/'
    prefix = prefix
    for root, _, files in os.walk(path):
        if '__init__.py' in files:
            packages.append(re.sub('^[^A-z0-9_]', '', root.replace('/', '.')))

    return packages


setup(
    name='PALM',
    version="0.1.0",
    description='PAML: PAddle Language Module',
    long_description=long_description,
    url='https://github.com/PaddlePaddle/models/PaddleNLP',
    packages=_find_packages(),
    package_data={'': ['*.so']},
    install_requires=["pyyaml>=3.12"],
    platforms=["Windows", "Mac", "Linux"],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ], )

if len(sys.argv) == 2 and sys.argv[1] == "install":
    #remove build/dict/eggs

    os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')

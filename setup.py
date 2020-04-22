"""
setup.py
"""

import os
from typing import Dict

from setuptools import find_packages, setup

NAME = "aum"
AUTHOR = "ASAPP Inc."
EMAIL = "jshapiro@asapp.com"
DESCRIPTION = "Library for calculating area under the margin ranking."


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


def required():
    with open('requirements.txt') as f:
        return f.read().splitlines()


# So that we don't import flambe.
VERSION: Dict[str, str] = {}
with open("aum/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name=NAME,
    version=os.environ.get("TAG_VERSION", VERSION['VERSION']),
    description=DESCRIPTION,

    # Author information
    author=AUTHOR,
    author_email=EMAIL,

    # What is packaged here.
    packages=find_packages(),
    install_requires=required(),
    include_package_data=True,
    python_requires='>=3.7.1',
    zip_safe=True)

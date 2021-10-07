# Copyright (C) 2021 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The setup.py file for python_graphs."""

from setuptools import setup

LONG_DESCRIPTION = """
python_graphs is a static analysis tool for performing control flow and data
flow analyses on Python programs, and for constructing Program Graphs.
Python Program Graphs are graph representations of Python programs suitable
for use with graph neural networks.
""".strip()

SHORT_DESCRIPTION = """
A library for generating graph representations of Python programs.""".strip()

DEPENDENCIES = [
    'absl-py',
    'astunparse',
    'gast',
    'networkx',
    'pygraphviz',
    'six',
]

TEST_DEPENDENCIES = [
]

VERSION = '1.2.3'
URL = 'https://github.com/google-research/python-graphs'

setup(
    name='python_graphs',
    version=VERSION,
    description=SHORT_DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,

    author='David Bieber',
    author_email='dbieber@google.com',
    license='Apache Software License',

    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',

        'Operating System :: OS Independent',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Operating System :: Unix',
    ],

    keywords='python program control flow data flow graph neural network',

    packages=['python_graphs'],

    install_requires=DEPENDENCIES,
    tests_require=TEST_DEPENDENCIES,
)

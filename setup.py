#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [ ]

test_requirements = ['pytest>=3', ]

setup(
    author="Hao Zhang",
    author_email='haozhang@alumni.princeton.edu',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Language Model Decomposition",
    entry_points={
        'console_scripts': [
            'lmd=lmd.cli:main',
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme,
    include_package_data=True,
    keywords='lmd',
    name='lmd',
    packages=find_packages(include=['lmd', 'lmd.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/haozhg/lmd',
    version='0.1.0',
    zip_safe=False,
)

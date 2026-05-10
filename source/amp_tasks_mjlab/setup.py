"""Installation script for the 'amp_tasks_mjlab' Python package.

Mirrors the layout of ``amp_tasks`` but targets mjlab as the simulator
backend. Importing the package registers the AMP variants of mjlab tasks
into :mod:`mjlab.tasks.registry`.
"""

from setuptools import find_packages, setup

PACKAGE_NAME = "amp_tasks_mjlab"
VERSION = "0.1.0"
DESCRIPTION = "AMP tasks running on the mjlab simulator backend."

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
    zip_safe=False,
)

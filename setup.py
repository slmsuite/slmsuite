"""
setup.py - this module makes the package installable
"""

from setuptools import setup, find_packages

NAME = "slmsuite"
VERSION = "0.3.0"
DEPENDENCIES = [
    "numpy",
    "scipy",
    "opencv-python",
    "matplotlib",
    "h5py",
    "tqdm"
]
DESCRIPTION = (
    "Package for high-performance spatial light "
    "modulator (SLM) control and holography."
)
AUTHOR = "slmsuite developers"
AUTHOR_EMAIL = "qp-slm@mit.edu"

setup(
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    install_requires=DEPENDENCIES,
    name=NAME,
    version=VERSION,
    packages=find_packages(),
)

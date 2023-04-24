"""
setup.py - this module makes the package installable
"""

from distutils.core import setup

NAME = "slmsuite"
VERSION = "0.0.1"
DEPENDENCIES = [
    "numpy",
    "scipy",
    "opencv-python",
    "matplotlib",
    "h5py",
    "tqdm"
]
DESCRIPTION = ("Package for high-performance spatial light "
               "modulator (SLM) control and holography.")
AUTHOR = "MIT Quantum Photonics Group"
AUTHOR_EMAIL = "qp-slm@mit.edu"

setup(author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      install_requires=DEPENDENCIES,
      name=NAME,
      version=VERSION,
)

.. _installation:

Installation
============

Install the stable version of |slmsuite|_ from `PyPi <https://pypi.org/project/slmsuite/>`_ using:

.. code-block:: console

    pip install slmsuite

Install the latest version of |slmsuite|_ from `GitHub <https://github.com/QPG-MIT/slmsuite>`_ using:

.. code-block:: console

    pip install git+https://github.com/QPG-MIT/slmsuite

Required Dependencies
---------------------

The following python packages are necessary to run |slmsuite|_. These are listed as PyPi
dependencies and thus are installed automatically if ``pip`` is used to install. One can
also use ``pip install -r requirements.txt`` to install these dependencies directly
without using ``pip`` to install |slmsuite|_.

- `python <https://www.python.org/>`_
- `numpy <https://numpy.org/>`_
- `scipy <https://scipy.org/>`_
- `opencv-python <https://github.com/opencv/opencv-python>`_
- `matplotlib <https://matplotlib.org/>`_
- `h5py <https://www.h5py.org/>`_
- `tqdm <https://github.com/tqdm/tqdm>`_

Hardware Dependencies
---------------------

The following python packages are optional acceleration or hardware requirements, which
the user can install selectively.

- `cupy <https://cupy.dev/>`_ (highly recommended for GPU-accelerated holography)
    - Installation via ``conda install -c conda-forge cupy`` is
    `recommended <https://docs.cupy.dev/en/stable/install.html>`_.
- Cameras
    - `instrumental-lib <https://github.com/mabuchilab/Instrumental>`_
    - `pymmcore <https://github.com/micro-manager/pymmcore>`_
    - `pypylon <https://github.com/basler/pypylon>`_
    - `PySpin <https://www.flir.com/products/spinnaker-sdk/>`_ (non-PyPi)
    - `thorlabs_tsi_sdk <https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam>`_ (non-PyPi)
    - `VimbaPython <https://github.com/alliedvision/VimbaPython>`_ (non-PyPi)
    - Other cameras are loaded directly via .dll.
- SLMs
    - `pyglet <https://pyglet.org/>`_
    - Other SLMs are loaded directly via .dll.

Jupyter
-------

We highly recommended using `Jupyter <https://jupyter.org>`_
notebooks for interactive computing,
and also list useful packages for code profiling which can be included via
`IPython <https://ipython.org/>`_
`magic <https://ipython.readthedocs.io/en/stable/interactive/tutorial.html#magics-explained>`_,
along with other features like |autoreload|_ or |matplotlibs|_ which are packaged with IPython.
To install recommended jupyter-related packages, use ``pip install -r requirements_ipython.txt``.

- `jupyter <https://jupyter.org>`_
    - `line-profiler <https://github.com/pyutils/line_profiler>`_
    - `snakeviz <https://github.com/jiffyclub/snakeviz>`_

If Jupyter is not used, the default :mod:`matplotlib` plots will block further
execution, so the user should avoid plotting with ``plot=False`` flags on functions
or develop a workaround.

.. |slmsuite| replace:: :mod:`slmsuite`
.. _slmsuite: https://github.com/QPG-MIT/slmsuite

.. |autoreload| replace:: ``%autoreload 2``
.. _autoreload: https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html

.. |matplotlibs| replace:: ``%matplotlib inline``
.. _matplotlibs: https://ipython.readthedocs.io/en/stable/interactive/plotting.html
.. _installation:

Installation
============

Install the latest version of |slmsuite|_ from `PyPi <http://google.com>`_ using:

.. code-block:: console

    pip install slmsuite

Required Dependencies
---------------------

The following packages are necessary to run |slmsuite|_. These are listed as PyPi
dependencies and thus are installed automatically if ``pip`` is used.

- `python <https://www.python.org/>`_
- `numpy <https://numpy.org/>`_
- `scipy <https://scipy.org/>`_
- `opencv-python <https://github.com/opencv/opencv-python>`_
- `matplotlib <https://matplotlib.org/>`_
- `h5py <https://www.h5py.org/>`_
- `tqdm <https://github.com/tqdm/tqdm>`_

Hardware Dependencies
---------------------

The following packages are optional acceleration or hardware requirements, which
the user can install selectively.

- `cupy <https://cupy.dev/>`_ (highly recommended for GPU-accelerated holography)
- Cameras
    - `VimbaPython <https://github.com/alliedvision/VimbaPython>`_ (non-PyPi)
    - `Xenics SDK <https://www.xenics.com/software/>`_ (non-PyPi)
    - `PySpin <https://www.flir.com/products/spinnaker-sdk/>`_ (non-PyPi)
    - `pymmcore <https://github.com/micro-manager/pymmcore>`_
    - `thorlabs_tsi_sdk <https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam>`_ (non-PyPi)
- SLMs
    - `pyglet <https://pyglet.org/>`_
    - `Santec <https://www.santec.com/en/products/components/slm/>`_ (non-PyPi)

Jupyter
-------

We highly recommended using `Jupyter <https://jupyter.org>`_
notebooks for interactive computing,
and also list useful packages for code profiling which can be included via
`IPython <https://ipython.org/>`_
`magic <https://ipython.readthedocs.io/en/stable/interactive/tutorial.html#magics-explained>`_,
along with other features like |autoreload|_ or |matplotlib|_ which are packaged with IPython.

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

.. |matplotlib| replace:: ``%matplotlib inline``
.. _matplotlib: https://ipython.readthedocs.io/en/stable/interactive/plotting.html
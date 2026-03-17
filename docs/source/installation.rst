.. _installation:

Installation
============

We recommend `uv <https://docs.astral.sh/uv/>`_ as a modern, fast Python package manager.
All commands below use ``uv``. If you are using ``pip`` instead, note that ``uv pip install``
installs packages in editable mode by default, whereas ``pip`` does not — you will need to
pass the ``-e`` flag explicitly (e.g. ``pip install -e .``) for an editable install.

PyPI
----

Install the stable version of |slmsuite|_ from `PyPI <https://pypi.org/project/slmsuite/>`_ using:

.. code-block:: console

    uv pip install slmsuite

GitHub
------

Install the latest version of |slmsuite|_ from `GitHub <https://github.com/holodyne/slmsuite>`_ using:

.. code-block:: console

    uv pip install git+https://github.com/holodyne/slmsuite

One can also clone |slmsuite|_ directly and add its directory to the Python path.
*Remember to install the dependencies (next sections)*.

.. code-block:: console

    git clone https://github.com/holodyne/slmsuite

Required Dependencies
---------------------

The following python packages are necessary to run |slmsuite|_. These are listed as PyPI
dependencies and thus are installed automatically if PyPI is used to install.

- `python <https://www.python.org/>`_
- `numpy <https://numpy.org/>`_
- `scipy <https://scipy.org/>`_
- `opencv-python <https://github.com/opencv/opencv-python>`_
- `matplotlib <https://matplotlib.org/>`_
- `h5py <https://www.h5py.org/>`_
- `tqdm <https://github.com/tqdm/tqdm>`_

One can also install these dependencies directly by calling the following inside
the package directory.

.. code-block:: console

    uv pip install .

Hardware Dependencies
---------------------

The following python packages are *optional* acceleration or hardware requirements, which
the user can install selectively.

- GPU ``uv pip install .[gpu]``
    - `cupy <https://cupy.dev/>`_, highly recommended for GPU-accelerated holography.
      Sometimes, installation is made complicated by a pre-installed version of CUDA.
      You can find the CUDA version with ``nvcc --version`` in a terminal, and then
      install an installation of :mod:`cupy` specific to CUDA version ``YY`` with
      ``uv pip install cupy-cudaYYx``.
- Gradients ``uv pip install .[torch]``
    - `pytorch <https://pytorch.org/>`_, required for conjugate gradient hologram
      optimization, either in GPU or CPU mode. Uses :mod:`cupy` - :mod:`torch`
      `interoperability <https://docs.cupy.dev/en/stable/user_guide/interoperability.html#pytorch>`_
      to pass data between modules without copying overhead, even on the GPU.
- Cameras ``uv pip install .[cameras]``
    - `instrumental-lib <https://github.com/mabuchilab/Instrumental>`_
    - `pylablib <https://github.com/AlexShkarin/pyLabLib>`_
    - `pymmcore <https://github.com/micro-manager/pymmcore>`_
    - `pypylon <https://github.com/basler/pypylon>`_
    - `mvsdk <https://www.mindvision.com.cn/category/software/demo-development-routine/>`_ (non-PyPI)
    - `PySpin <https://www.flir.com/products/spinnaker-sdk/>`_ (non-PyPI)
    - `tisgrabber <https://github.com/TheImagingSource/IC-Imaging-Control-Samples/tree/master/Python/tisgrabber>`_ (non-PyPI)
    - `thorlabs_tsi_sdk <https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam>`_ (non-PyPI)
    - `VmbPy <https://github.com/alliedvision/VmbPy>`_ (non-PyPI)
    - Other cameras are loaded directly via .dll.
- SLMs ``uv pip install .[slms]``
    - `pyglet <https://pyglet.org/>`_
    - Other SLMs are loaded directly via .dll.
- Image saving ``uv pip install .[images]``
    - For most images and videos, `imageio <https://imageio.readthedocs.io/en/stable/>`_
    - Many video formats additionally require `pyav <https://pypi.org/project/av/>`_
    - For .gif optimization, `pygifsicle <https://pypi.org/project/pygifsicle/>`_

Jupyter
-------

We highly recommended using `Jupyter <https://jupyter.org>`_
notebooks for interactive computing. Consider also using
`IPython <https://ipython.org/>`_
`magic <https://ipython.readthedocs.io/en/stable/interactive/tutorial.html#magics-explained>`_,
features like |autoreload|_ or |matplotlibs|_.

- `jupyter <https://jupyter.org>`_

If Jupyter is not used, the default :mod:`matplotlib` plots will block further
execution, so the user should avoid plotting by using ``plot=False`` flags on functions.

Use the following to install recommended jupyter-related packages.

.. code-block:: console

    uv pip install .[jupyter]


.. |slmsuite| replace:: :mod:`slmsuite`
.. _slmsuite: https://github.com/holodyne/slmsuite

.. |autoreload| replace:: ``%autoreload 2``
.. _autoreload: https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html

.. |matplotlibs| replace:: ``%matplotlib inline``
.. _matplotlibs: https://ipython.readthedocs.io/en/stable/interactive/plotting.html
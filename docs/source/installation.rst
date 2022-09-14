.. _installation:

Installation
============

Install the latest version of |slmsuite|_ from PyPi using:

.. code-block:: console

    pip install slmsuite

Required Dependencies
---------------------

- `python <https://www.python.org/>`_
- `numpy <https://numpy.org/>`_
- `scipy <https://scipy.org/>`_
- `opencv-python <https://github.com/opencv/opencv-python>`_
- `matplotlib <https://matplotlib.org/>`_
- `h5py <https://www.h5py.org/>`_
- `tqdm <https://github.com/tqdm/tqdm>`_

Hardware Dependencies
---------------------

- `cupy <https://cupy.dev/>`_ (highly recommended for GPU-accelerated holography)
- Cameras
    - `vimba <https://github.com/alliedvision/VimbaPython>`_ (non-PyPi)
    - `Xenics SDK <https://www.xenics.com/software/>`_ (non-PyPi)
    - `PySpin <https://www.flir.com/products/spinnaker-sdk/>`_ (non-PyPi)
    - `pymmcore <https://github.com/micro-manager/pymmcore>`_
    - `thorlabs_tsi_sdk <https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=ThorCam>`_ (non-PyPi)
- SLMs
    - `pyglet <https://pyglet.org/>`_
    - `Santec <https://www.santec.com/en/products/components/slm/>`_ (non-PyPi)

.. |slmsuite| replace:: :mod:`slmsuite`
.. _slmsuite: https://github.com/QPG-MIT/slmsuite
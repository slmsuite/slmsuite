.. _why:

Why :mod:`slmsuite`?
====================

TODO @cpanuski

Why :mod:`python`?
------------------

.. Easy and accessible to scientists.

Python is a simple and fast programming language [1]_.
With easy-to-learn syntax and good readability, python is ideal for scientists and
experimentalists interested in quick development and refininement.
Itself being open-source [2]_, python is a goto language for open-source projects,
to the point that it is the most pull-requested language on GitHub [3]_.

.. Fast and hardware-compatible due to C backend.

The default implementation of the python language, :mod:`cpython` [2]_, is built
on top of C.
Scientific computing packages such as :mod:`numpy` and :mod:`scipy` implement
algorithms in fast C code, leading to a paradigm where 'heavy lifting' is done in C,
while higher-level logic or 'heavy coding' is done in the more user-friendly python.
Importantly, the accessiblity of the C backend means that hardware interfaces
(often written in C) are easy to implement in python [4]_. This is critical for
the :mod:`slmsuite` package, with a focus on experimental holography using physical
cameras and SLMs.

.. jupyter is cool too.

We find jupyter notebooks with autoreload [5]_ to be exceptionally useful for
experimentation and debugging, as this produces MATLAB-like interactivity with
editable scripting.
See the jupyter notebooks in :ref:`examples` to try it out!

Why :mod:`cupy`?
----------------

.. Even faster with a GPU!

Core :mod:`~slmsuite.holography.algorithms` in :mod:`slmsuite` make heavy use of
fast Fourier transforms on large arrays.
These problems are ideal for GPU-based acceleration, implemented here with :mod:`cupy` [6]_.
In most cases, :mod:`cupy` is a drop-in replacement for :mod:`numpy` and :mod:`scipy` [7]_,
which are used as a backup if a GPU is not present.
We repeatably measure around two orders of magnitude speedup for common operation
when using :mod:`cupy`, compared with the :mod:`numpy` equivalent.

References
----------

.. [1] https://www.python.org/
.. [2] https://github.com/python/cpython
.. [3] https://madnight.github.io/githut/#/pull_requests/
.. [4] https://docs.python.org/3/library/ctypes.html
.. [5] https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html
.. [6] https://cupy.dev/
.. [7] https://docs.cupy.dev/en/stable/reference/comparison.html
.. _why:

Why :mod:`slmsuite`?
====================

TODO @cpanuski

Why :mod:`python`?
------------------

.. Easy and accessible to scientists.

`Python <https://www.python.org/>`_ is a simple and fast programming language.
With easy-to-learn syntax and good readability, python is ideal for scientists and
experimentalists interested in quick development and refininement.
Itself being `open-source <https://github.com/python/cpython>`_,
python is a goto language for open-source projects, to the point that it is the
`most pull-requested language <https://madnight.github.io/githut/#/pull_requests/>`_
on GitHub.

.. Fast and hardware-compatible due to C backend.

The default implementation of the python language,
`cpython <https://github.com/python/cpython>`_, is built on top of C.
Scientific computing packages such as |numpy|_ and |scipy|_ implement
algorithms in fast C code, leading to a paradigm where 'heavy lifting' is done in C,
while higher-level logic or 'heavy coding' is done in the more user-friendly python.
Importantly, the accessiblity of the C backend means that hardware interfaces
(often written in C) are
`easy to implement <https://docs.python.org/3/library/ctypes.html>`_
in python. This is critical for
the |slmsuite|_ package, with a focus on experimental holography using physical
cameras and SLMs.

.. jupyter is cool too.

We find jupyter notebooks with
`autoreload <https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html>`_
to be exceptionally useful for
experimentation and debugging, as this produces MATLAB-like interactivity with
editable scripting.
See the jupyter notebooks in :ref:`examples` to try it out!

Why :mod:`cupy`?
----------------

.. Even faster with a GPU!

Core :mod:`~slmsuite.holography.algorithms` in |slmsuite|_ make heavy use of
fast Fourier transforms on large arrays. These problems are ideal for GPU-based
acceleration, implemented here with |cupy|_. In most cases, |cupy|_ is a
`drop-in replacement <https://docs.cupy.dev/en/stable/reference/comparison.html>`_
for |numpy|_ and |scipy|_, which are used as a backup if a GPU is not present.
We repeatably measure around two orders of magnitude speedup for standard
optimization when using |cupy|_, compared with the |numpy|_ equivalent.

.. Linked modules

.. |numpy| replace:: :mod:`numpy`
.. _numpy: https://numpy.org/

.. |scipy| replace:: :mod:`scipy`
.. _scipy: https://scipy.org/

.. |cupy| replace:: :mod:`cupy`
.. _cupy: https://cupy.dev/

.. |slmsuite| replace:: :mod:`slmsuite`
.. _slmsuite: https://github.com/QPG-MIT/slmsuite
"""
Projects data onto the SLM's virtual display, using the :mod:`pyglet` library.
"""
import warnings
import numpy as np

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware._pyglet import _Window

try:
    import pyglet
except ImportError:
    pyglet = None
    warnings.warn("pyglet not installed. Install to use ScreenMirrored SLMs.")

class ScreenMirrored(SLM):
    """
    Wraps a :mod:`pyglet` window for displaying data to an SLM.

    Important
    ~~~~~~~~~
    Many SLM manufacturers provide an SDK for interfacing with their devices.
    Using a python wrapper for these SDKs is recommended, instead of or in supplement to this class,
    as there often is functionality additional to a mirrored screen
    (e.g. USB for changing settings) along with device-specific optimizations.

    Note
    ~~~~
    There are a variety of python packages that support blitting images onto a fullscreen display.

    -   `Simple DirectMedia Layer (SDL) <https://www.libsdl.org/>`_ wrappers:

        - :mod:`pygame` (`link <https://www.pygame.org/docs/>`__),
          which also supports OpenGL. Only supports one screen.
        - :mod:`sdl2` (`readthedocs <https://pysdl2.readthedocs.io/en/latest/>`__)
          through the ``PySDL2`` package. Requires additional libraries.

    -   `Open Graphics Library (OpenGL) <https://www.opengl.org/>`_ wrappers:

        - :mod:`moderngl` (`readthedocs <https://moderngl.readthedocs.io/en/latest/>`__),
          an OpenGL wrapper focusing on a pythonic interface for core OpenGL functions.
        - :mod:`OpenGL` (`link <http://pyopengl.sourceforge.net/documentation/index.html>`__)
          through the ``PyOpenGL``/``PyOpenGL_accelerate`` package, a very light OpenGL wrapper.
        - :mod:`pyglet` (`readthedocs <https://pyglet.readthedocs.io/en/latest/>`__),
          a light OpenGL wrapper.

    -   GUI Library wrappers:

        - :mod:`gi` (`readthedocs <https://pygobject.readthedocs.io/en/latest/>`__),
          through the ``PyGObject`` package wrapping ``GTK`` and other GUI libraries.
        - :mod:`pyqt6` (`link <https://riverbankcomputing.com/software/pyqt/>`__),
          through the ``PyQt6`` package wrapping the version 6 ``Qt`` GUI library.
        - :mod:`tkinter` (`link <https://docs.python.org/3/library/tkinter.html>`__),
          included in standard ``python``, wrapping the ``Tcl``/``Tk`` GUI library.
        - :mod:`wx` (`link <https://docs.wxpython.org/>`__),
          through the ``wxPython`` package wrapping the ``wxWidgets`` GUI library.
          :mod:`slmpy` (`GitHub <https://github.com/wavefrontshaping/slmPy>`__) uses :mod:`wx`.

    :mod:`slmsuite` uses :mod:`pyglet` as the default display package.
    :mod:`pyglet` is generally more capable than the mentioned SDL wrappers while immediately supporting
    features such as detecting connected displays which low-level packages like :mod:`OpenGL` and
    :mod:`moderngl` do not have. :mod:`pyglet` allows us to interact more directly with the display
    hardware without the additional overhead that is found in GUI libraries.
    Most importantly, :mod:`pyglet` is well documented.

    However, it might be worthwhile in the future to look back into SDL options, as SDL surfaces
    are closer to the pixels than OpenGL textures, so greater speed might be achievable (even without
    loading data to the GPU as a texture). Another potential improvement could come from writing
    :mod:`cupy` datastructures to ``OpenGL`` textures directly, without using the CPU as an
    intermediary. There is `some precedent <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html>`_
    for transferring data from ``CUDA`` (on which :mod:`cupy` is based) to ``OpenGL``,
    though :mod:`cupy` does not currently directly support this.

    Important
    ~~~~~~~~~
    :class:`ScreenMirrored` uses a double-buffered and vertically synchronized (vsync) ``OpenGL``
    context. This is to prevent "tearing" resulting from data being modified during a display write:
    rather, all monitor writes are synchronized such that clean frames are always displayed.
    This feature is similar to the ``isImageLock`` flag in :mod:`slmpy`, but is implemented a bit
    closer to the hardware.

    Attributes
    ----------
    window : pyglet.window.Window
        Fullscreen window used to send information to the SLM.
    tex_shape_ratio : (int, int)
        Ratio between the SLM shape and the (power-2-padded) texture stored in ``OpenGL``.
    buffer : numpy.ndarray
        Memory used to load data to the ``OpenGL`` memory. Of type ``np.uint8``.
    cbuffer : pyglet.gl.GLubyte
        Array of length ``prod(shape) * 4`` (4 bytes per RGBA).
        Maps to the same memory as :attr:`buffer`.
        Used to load data to the texture.
    texture : pyglet.gl.GLuint
        Identifier for the texture loaded into ``OpenGL`` memory.
    """

    def __init__(
            self,
            display_number,
            bitdepth=8,
            wav_um=1,
            pitch_um=(8,8),
            verbose=True,
            **kwargs
        ):
        """
        Initializes a :mod:`pyglet` window for displaying data to an SLM.

        Caution
        ~~~~~~~
        An SLM designed at 1064 nm can be used for an application at 780 nm by passing
        ``wav_um=.780`` and ``wav_design_um=1.064``,
        thus causing the SLM to use only a fraction (780/1064)
        of the full dynamic range. Be sure these values are correct.
        Note that there are some performance losses from using this modality (see :meth:`.set_phase()`).

        Caution
        ~~~~~~~
        There is some subtlety to
        `complex display setups with Linux <https://pyglet.readthedocs.io/en/latest/modules/canvas.html>`_.
        Working outside the default display is currently not implemented.

        Parameters
        ----------
        display_number : int
            Monitor number for frame to be instantiated upon.
        bitdepth : int
            Bitdepth of the SLM. Defaults to 8.

            Caution
            ~~~~~~~
            This class currently supports SLMs with 8-bit precision or less.
            In the future, this class will also support 16-bit SLMs using RG color.
        wav_um : float
            Wavelength of operation in microns. Defaults to 1 um.
        pitch_um : (float, float)
            Pixel pitch in microns. Defaults to 8 micron square pixels.
        verbose : bool
            Whether or not to print extra information.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.
        """
        if pyglet is None:
            raise ImportError("pyglet not installed. Install to use ScreenMirrored SLMs.")

        if verbose:
            print("Initializing pyglet... ", end="")
        display = pyglet.canvas.get_display()
        screens = display.get_screens()
        if verbose:
            print("success")
            print("Searching for window with display_number={}... "
                    .format(display_number), end="")

        if len(screens) <= display_number:
            raise ValueError("Could not find display_number={}; only {} displays"
                .format(display_number, len(screens)))

        screen_info = ScreenMirrored.info(verbose=False)

        if screen_info[display_number][3]:
            raise ValueError(
                "ScreenMirrored window already created on display_number={}"
                .format(display_number))

        if verbose and screen_info[display_number][2]:
            print("warning: this is the main display... ", end="")

        if verbose:
            print("success")
            print("Creating window... ", end="")

        screen = screens[display_number]

        super().__init__(
            (screen.width, screen.height),
            bitdepth=bitdepth,
            wav_um=wav_um,
            pitch_um=pitch_um,
            **kwargs
        )

        self.window = _Window(None, screen, self.name)

        try:
            self.window._setup_context()
        except Exception as e:
            print("failure")
            self.window.close()
            raise e

        if verbose:
            print("success")

        # Warn the user if wav_um > wav_design_um
        if self.phase_scaling > 1:
            print(
                "Warning: Wavelength {} um is inaccessible to this SLM with "
                "design wavelength {} um".format(self.wav_um, self.wav_design_um)
            )

    def _set_phase_hw(self, data):
        """Writes to screen. See :class:`.SLM`."""
        # Write to buffer (.buffer points to the same data as .cbuffer).
        # Unfortunately, OpenGL2.0 needs the data copied three times (I think).
        # FUTURE: For OpenGL3.0 and pyglet 2.0+, use the shader to minimize data transfer.
        np.copyto(self.window.buffer[:,:,0], data)
        np.copyto(self.window.buffer[:,:,1], data)
        np.copyto(self.window.buffer[:,:,2], data)

        self.window.render()

    def close(self):
        """Closes frame. See :class:`.SLM`."""
        self.window.close()

    @staticmethod
    def info(verbose=True):
        """
        Get information about the available displays, their indexes, and their sizes.

        Parameters
        ----------
        verbose : bool
            Whether or not to print display information.

        Returns
        -------
        list of (int, (int, int, int, int), bool, bool) tuples
            The number, geometry of each display.
        """
        if pyglet is None:
            raise ImportError("pyglet not installed. Install to use ScreenMirrored SLMs.")

        return _Window.info(verbose=verbose)
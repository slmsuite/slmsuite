"""
An SLM mirrored onto a display, using the :mod:`pyglet` library.
"""
import numpy as np
import ctypes
import os
from slmsuite.hardware.slms.slm import SLM

try:
    import pyglet
    import pyglet.gl as gl
except ImportError:
    print("screenmirrored.py: pyglet not installed. Install to use ScreenMirrored SLMs.")


class ScreenMirrored(SLM):
    """
    Wraps a :mod:`pyglet` window for displaying data to an SLM.

    Caution
    ~~~~~~~
    This class currently supports SLMs with 8-bit precision or less.
    In the future, this class will support 16-bit SLMs using RG color.

    Important
    ~~~~~~~~~
    Many SLM manufacturers provide an SDK for interfacing with their devices.
    Using a python wrapper for these SDKs is recommended, instead of or in supplement to this class,
    as there often is functionality additional to a mirrored screen
    (e.g. USB for changing settings) along with device-specific optimizations.

    Note
    ~~~~
    There are a variety of python packages that support blitting images onto a fullscreen display.

    - `Simple DirectMedia Layer (SDL) <https://www.libsdl.org/>`_ wrappers:

        - :mod:`pygame` (`link <https://www.pygame.org/docs/>`_),
          which also supports OpenGL. Only supports one screen.
        - :mod:`sdl2` (`readthedocs <https://pysdl2.readthedocs.io/en/latest/>`_)
          through the ``PySDL2`` package. Requires additional libraries.

    - `Open Graphics Library (OpenGL) <https://www.opengl.org/>`_ wrappers:

        - :mod:`moderngl` (`readthedocs <https://moderngl.readthedocs.io/en/latest/>`_),
          an OpenGL wrapper focusing on a pythonic interface for core OpenGL functions.
        - :mod:`OpenGL` (`link <http://pyopengl.sourceforge.net/documentation/index.html>`_)
          through the ``PyOpenGL``/``PyOpenGL_accelerate`` package, a very light OpenGL wrapper.
        - :mod:`pyglet` (`readthedocs <https://pyglet.readthedocs.io/en/latest/>`_),
          a light OpenGL wrapper.

    - GUI Library wrappers:

        - :mod:`gi` (`readthedocs <https://pygobject.readthedocs.io/en/latest/>`_),
          through the ``PyGObject`` package wrapping ``GTK`` and other GUI libraries.
        - :mod:`pyqt6` (`link <https://riverbankcomputing.com/software/pyqt/>`_),
          through the ``PyQt6`` package wrapping the version 6 ``Qt`` GUI library.
        - :mod:`tkinter` (`link <https://docs.python.org/3/library/tkinter.html>`_),
          included in standard ``python``, wrapping the ``Tcl``/``Tk`` GUI library.
        - :mod:`wx` (`link <https://docs.wxpython.org/>`_),
          through the ``wxPython`` package wrapping the ``wxWidgets`` GUI library.
          :mod:`slmpy` (`GitHub <https://github.com/wavefrontshaping/slmPy>`_) uses :mod:`wx`.

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

    def __init__(self, display_number, bitdepth=8, verbose=True, **kwargs):
        """
        Initializes a :mod:`pyglet` window for displaying data to an SLM.

        Caution
        ~~~~~~~
        :mod:`slmsuite` makes use of user-supplied knowledge
        of SLM pixel size: :attr:`.SLM.dx_um`, :attr:`.SLM.dy_um`.
        Be sure these values are correct.

        Caution
        ~~~~~~~
        An SLM designed at 1064 nm can be used for an application at 780 nm by passing
        ``wav_um=.780`` and ``wav_design_um=1.064``,
        thus causing the SLM to use only a fraction (780/1064)
        of the full dynamic range. Be sure these values are correct.
        Note that there are some performance losses from using this modality (see :meth:`.write()`).

        Caution
        ~~~~~~~
        There is some subtlety to
        `complex display setups with Linux <https://pyglet.readthedocs.io/en/latest/modules/canvas.html>`_.
        Working outside the default display is currently not implemented.

        Parameters
        ----------
        display_number : int
            Monitor number for frame to be instantiated upon.
        verbose : bool
            Whether or not to print extra information.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.

        References
        ----------
        .. [14]
        """
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
            print("warning: this is the main display... ")

        if verbose:
            print("success")
            print("Creating window... ", end="")

        screen = screens[display_number]

        super().__init__(screen.width, screen.height, bitdepth=bitdepth, **kwargs)

        # Setup the window. If failure, closes elegantly upon except().
        try:
            # Make the window and do basic setup.
            self.window = pyglet.window.Window( screen=screen,
                                                fullscreen=True, vsync=True)
            self.window.set_caption(self.name)
            self.window.set_mouse_visible(False)

            try:
                # Icons. Currently hardcoded. Feel free to implement custom icons.
                path, filename = os.path.split(os.path.realpath(__file__))
                path = os.path.join(path, '..', '..', '..',
                                    'docs', 'source', 'static', 'qp-slm-notext-')
                img16x16 = pyglet.image.load(path + '16x16.png')
                img32x32 = pyglet.image.load(path + '32x32.png')
                self.window.set_icon(img16x16, img32x32)
            except Exception as e:
                print(e)

            # Set the viewpoint.
            proj = pyglet.window.Projection2D()
            proj.set(self.shape[1], self.shape[0], self.shape[1], self.shape[0])

            # Setup shapes
            texture_shape = tuple(np.power(2, np.ceil(np.log2(self.shape)))
                                .astype(np.int64))
            self.tex_shape_ratio = (float(self.shape[0])/float(texture_shape[0]),
                                    float(self.shape[1])/float(texture_shape[1]))
            B = 4

            # Setup buffers (texbuffer is power of 2 padded to init the memory in OpenGL)
            self.buffer = np.zeros(self.shape + (B,), dtype=np.uint8)
            N = int(self.shape[0] * self.shape[1] * B)
            self.cbuffer = (gl.GLubyte * N).from_buffer(self.buffer)

            texbuffer = np.zeros(texture_shape + (B,), dtype=np.uint8)
            Nt = int(texture_shape[0] * texture_shape[1] * B)
            texcbuffer = (gl.GLubyte * Nt).from_buffer(texbuffer)

            # Setup the texture
            gl.glEnable(gl.GL_TEXTURE_2D)
            self.texture = gl.GLuint()
            gl.glGenTextures(1, ctypes.byref(self.texture))
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.value)

            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_GENERATE_MIPMAP, gl.GL_FALSE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

            # Malloc the OpenGL memory
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8,
                            texture_shape[1], texture_shape[0],
                            0, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE,
                            texcbuffer)

            # Make sure we can write to a subset of the memory (as we will do in the future)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0,
                            self.shape[1], self.shape[0],
                            gl.GL_BGRA, gl.GL_UNSIGNED_BYTE,
                            self.cbuffer)

            # Cleanup
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glFlush()

            # Write nothing.
            self.write(phase=None)
        except:
            try:
                self.window.close()
            except:
                pass
            raise

        if verbose:
            print("success")

        # Warn the user if wav_um > wav_design_um
        if self.phase_scaling > 1:
            print(
                "Warning: Wavelength {} um is inaccessible to this SLM with "
                "design wavelength {} um".format(self.wav_um, self.wav_design_um)
            )

    def _write_hw(self, data):
        """Writes to screen. See :class:`.SLM`."""
        # Write to buffer (self.buffer is the same as self.cbuffer).
        # Unfortunately, OpenGL needs the data copied three times.
        np.copyto(self.buffer[:,:,0], data)
        np.copyto(self.buffer[:,:,1], data)
        np.copyto(self.buffer[:,:,2], data)

        # Setup texture variables.
        x1 = 0
        y1 = 0
        x2 = self.shape[1]
        y2 = self.shape[0]

        xa = 0
        ya = 0
        xb = self.tex_shape_ratio[1]
        yb = self.tex_shape_ratio[0]

        array = (gl.GLfloat * 32)(
            xa, ya, 0., 1.,         # tex coord,
            x1, y1, 0., 1.,         # real coord, ...
            xb, ya, 0., 1.,
            x2, y1, 0., 1.,
            xb, yb, 0., 1.,
            x2, y2, 0., 1.,
            xa, yb, 0., 1.,
            x1, y2, 0., 1.)

        # Update the texture.
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.value)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0,
                        self.shape[1], self.shape[0],
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                        self.cbuffer)

        # Blit the texture.
        gl.glPushClientAttrib(gl.GL_CLIENT_VERTEX_ARRAY_BIT)
        gl.glInterleavedArrays(gl.GL_T4F_V4F, 0, array)
        gl.glDrawArrays(gl.GL_QUADS, 0, 4)
        gl.glPopClientAttrib()

        # Display the other side of the double buffer.
        # (with vsync enabled, this will block until the next frame is ready to display).
        self.window.flip()

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
        list of (int, (int, int, int, int)) tuples
            The number and geometry of each display.
        """
        # Note: in pyglet, the display is the full arrangement of screens,
        # unlike the terminology in other SLM subclasses
        display = pyglet.canvas.get_display()

        screens = display.get_screens()
        default = display.get_default_screen()
        windows = display.get_windows()

        def parse_screen(screen):
            return ("x={}, y={}, width={}, height={}"
                .format(screen.x, screen.y, screen.width, screen.height))
        def parse_screen_int(screen):
            return (screen.x, screen.y, screen.width, screen.height)
        def parse_window(window):
            x, y = window.get_location()
            return ("x={}, y={}, width={}, height={}"
                .format(x, y, window.width, window.height))

        default_str = parse_screen(default)

        window_strs = []
        for window in windows:
            window_strs.append(parse_window(window))

        if verbose:
            print('Display Positions:')
            print('#,  Position')

        screen_list = []

        for x, screen in enumerate(screens):
            screen_str = parse_screen(screen)

            main_bool = False
            window_bool = False

            if screen_str == default_str:
                main_bool = True
                screen_str += ' (main)'
            if screen_str in window_strs:
                window_bool = True
                screen_str += ' (has ScreenMirrored)'

            if verbose:
                print('{},  {}'.format(x, screen_str))

            screen_list.append((x, parse_screen_int(screen),
                                main_bool, window_bool))

        return screen_list

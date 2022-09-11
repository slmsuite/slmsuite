"""
An SLM mirrored onto a display, using the :mod:`wxWidgets` library [1]_ [2]_.

Inspired by slmPy [3]_.

References
----------
.. [1] https://wxpython.org/Phoenix/docs/html/index.html
.. [2] https://github.com/wxWidgets/Phoenix
.. [3] https://github.com/wavefrontshaping/slmPy
"""
import numpy as np
import time

from slmsuite.hardware.slms.slm import SLM

try:
    _WX_APP_INIT_TIME_S = 5  # s

    import wx

    class _SLMWindow(wx.Window):
        """
        SLM window.

        Attributes
        ----------
        res : tuple of ints
            The size of the display in pixels.
        img : wx.Image
            The image to display.
        _buffer : wx.Bitmap
            For storing the image to display.
        cursor : wx.Cursor
            Cursor for the window.
        """

        def __init__(self, *args, **kwargs):
            """See :meth:`wx.Window.__init__`."""
            self.res = kwargs.pop("res")
            kwargs["style"] = (
                kwargs.setdefault("style", wx.NO_FULL_REPAINT_ON_RESIZE)
                | wx.NO_FULL_REPAINT_ON_RESIZE
            )
            super().__init__(*args, **kwargs)

            # Hide cursor
            self.cursor = wx.Cursor(wx.CURSOR_BLANK)
            self.SetCursor(self.cursor)

            self.img = wx.Image(*self.res)
            self._buffer = wx.Bitmap(*self.res)

            self.Bind(wx.EVT_PAINT, self.on_paint)

        def on_paint(self, event):
            """See :meth:`wx.Window.Bind`."""
            self._buffer = self.img.ConvertToBitmap()
            wx.BufferedPaintDC(self, self._buffer)

        def update_image_direct(self, img):
            """
            Directly update :attr:`img`.

            Parameters
            ----------
            img
                See :attr:`img`.
            """
            self.img = img
            self.Refresh()
            self.Update()


    class _SLMFrame(wx.Frame):
        """
        SLM frame.
        """

        def __init__(self, display_number, always_top):
            """
            Initialize frame.

            Parameters
            ----------
            display_number
                See :meth:`set_monitor`.
            always_top
                See :const:`wx.STAY_ON_TOP`.
            """
            style = wx.DEFAULT_FRAME_STYLE
            if always_top:
                style = style | wx.STAY_ON_TOP
            self.set_monitor(display_number)
            super().__init__(
                None,
                -1,
                "SLM",
                pos=(self._x0, self._y0),
                size=(self._resX, self._resY),
                style=style,
            )
            self.window = _SLMWindow(self, res=(self._resX, self._resY))
            self.Show()
            self.ShowFullScreen(not self.IsFullScreen(), wx.FULLSCREEN_ALL)
            self.SetFocus()

        def set_monitor(self, display_number):
            """
            Select the display for this frame.

            Parameters
            ----------
            display_number
                See :class:`wx.Display`.

            Raises
            ------
            ValueError
                If `display_number` is invalid.
            """
            if display_number < 0 or display_number > wx.Display.GetCount() - 1:
                raise ValueError(
                    "Invalid display_number (display_number %d)." % display_number
                )
            self._x0, self._y0, self._resX, self._resY = wx.Display(
                display_number
            ).GetGeometry()

        def update_image_direct(self, img):
            """
            Directly update the image displayed on the window.

            Parameters
            ----------
            img
                See :attr:`SLMWindow.img`.
            """
            self.window.update_image_direct(img)

        def close(self):
            """
            Close frame and associated resources.
            """
            wx.CallAfter(self.Destroy)
except ImportError as e:
    print("wx not installed. Install to use wx-based ScreenMirrored SLM.")

class ScreenMirrored(SLM):
    """
    Wraps a wx frame for displaying data to the SLM. Currently only supports 8-bit SLMs.

    Attributes
    ----------
    app : wx.App
        :class:`wx.App` context.
    frame : SLMframe
        :class:`wx.Frame` to blit upon.
    """

    def __init__(self, display_number, wav_um, wav_design_um, dx_um, dy_um, bitdepth=8, verbose=True, **kwargs):
        """
        Initializes ``wx`` :attr:`frame`. See superclass :class:`.SLM`.

        Caution
        ~~~~~~~
        :mod:`slmsuite` makes use of user-supplied knowledge
        of SLM pixel size: :attr:`.SLM.dx_um`, :attr:`.SLM.dy_um`.
        Be sure these values are correct.

        Caution
        ~~~~~~~
        This class currently supports SLMs with 8-bit precision or less.
        In the future, this class will support 16-bit SLMs using RG color.

        Caution
        ~~~~~~~
        An SLM designed at 1064 nm can be used for an application at 780 nm by passing
        ``wav_um=.780`` and ``wav_design_um=1.064``,
        thus causing the SLM to use only a fraction (780/1064)
        of the full dynamic range. Be sure these values are correct.
        Note that there are some performance losses from using this modality (see :meth:`.write()`).


        Parameters
        ----------
        display_number : int
            Monitor number for frame to be instantiated upon.
        wav_design_um : float
            See superclass :class:`.SLM`. Useful for scaling the dynamic range of the SLM
            such that operation at different wavelengths can be achieved.
        dx_um, dy_um : float
            See superclass :class:`.SLM`.
        **kwargs
            See superclass :class:`.SLM` for permissible options.
        """
        # Initialize wx app.
        self.open_app(verbose=verbose)

        # Open frame.
        if verbose:
            print("Searching for display_number={}... ".format(display_number), end="")
        self.frame = _SLMFrame(display_number=display_number, always_top=True)
        if verbose:
            print("success")

        # Super initialization.
        if verbose:
            print(
                "Initializing display with geometry={}... "
                "".format(wx.Display(display_number).GetGeometry()),
                end="",
            )
        super().__init__(
            self.frame._resX, self.frame._resY, bitdepth=bitdepth, **kwargs
        )
        if verbose:
            print("success")
    def _write_hw(self, phase):
        """See :meth:`SLM._write_hw`."""
        bw_array = phase.astype(np.uint8)
        bw_array.shape = self.shape[0], self.shape[1], 1
        self.frame.update_image_direct(
            wx.ImageFromBuffer(
                width=self.shape[1],
                height=self.shape[0],
                dataBuffer=np.concatenate(
                    (bw_array, bw_array, bw_array), axis=2
                ).tobytes(),
            )
        )

    @classmethod
    def open_app(cls, verbose=True):
        """
        Initialize a :class:`wx.App` instance for :attr:`app`.

        Parameters
        ----------
        verbose : bool
            Whether or not to print extra information.
        """
        if cls.app is None:
            if verbose:
                print("Starting wx... ", end="")
            cls.app = wx.App()
            time.sleep(_WX_APP_INIT_TIME_S)
            if verbose:
                print("success")

    @classmethod
    def close_app(cls):
        """Close :attr:`app`."""
        del cls.app
        cls.app = None

    def close(self, close_app=False):
        """
        Close :attr:`frame`.

        Parameters
        ----------
        close_app : bool
            See :meth:`close_app`.
        """
        self.frame.close()
        if close_app:
            self.close_app()

    @staticmethod
    def info(cls, verbose=True):
        """
        Discovers the positions of all the displays.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Returns
        --------
        list of (int, (int, int, int, int)) tuples
            The number and geometry of each display.
        """
        cls.open_app(verbose=verbose)
        display_count = wx.Display.GetCount()
        displays = list()
        for display_idx in range(display_count):
            display = wx.Display(display_idx)
            display_geometry = display.GetGeometry()
            displays.append((display_idx, display_geometry))
            if verbose:
                print("display {}: {}".format(display_idx, display_geometry))

        return displays
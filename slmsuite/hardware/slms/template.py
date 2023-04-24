"""
Template for writing a subclass for SLM hardware control.
Outlines which SLM super-class functions must be implemented. 
"""
from .slm import SLM

class Template(SLM):
    """
    Template for implementing a new SLM subclass. Replace ``Template``
    with the desired subclass name. :class:`~slmsuite.hardware.slms.SLM` is the
    superclass that sets the requirements for ``Template``.
    """

    def __init__(self, width, height, wav_um, pitch_um, bitdepth, **kwargs):
        r"""
        Instantiates the ``Template`` class  

        Arguments
        ---------

        width : int
            Width of the SLM in pixels.
        height : int
            Height of the SLM in pixels.
        wav_um : float
            Wavelength of operation in microns.
        pitch_um : float
            Pitch of SLM pixels in microns.
        bitdepth : int
            Bits of phase resolution (e.g. 8 for 256 phase settings.)
        kwargs
            See :meth:`.SLM.__init__` for permissible options.
            
        Note
        ~~~~
        These arguments, which ultimately are used to instatiate the :class:`.SLM` superclass,
        may be more accurately filled by calling the SLM's SDK functions.
        See the other implemented SLM subclasses for examples. 
        """

        # TODO: Insert code here to initialize the SLM hardware, load properties, etc.

        # Mandatory functions:
        # - Opening a connection to the device

        # Other possibilities to consider:
        # - Setting the SLMs hardware's operating wavelength
        # - Upsdating the SLM's phase table if necessary.
        # - Checking for and saving the SLM parameters (height, width, etc.)

        # Instantiate the superclass 
        super().__init__(
            width,
            height,
            bitdepth=bitdepth,
            wav_um=wav_um,
            dx_um=pitch_um,
            dy_um=pitch_um,
            **kwargs
        )

        # Zero the display using the super-class `write()` function.
        self.write(None)
        

    def _write_hw(self, phase):
        """Low-level hardware interface to write ``phase`` data onto the SLM.
        When the user calls the :meth:`.SLM.write` method of
        :class:`.SLM`, ``phase`` is error checked before calling
        :meth:`_write_hw()`. See :meth:`.SLM._write_hw` for further detail.
        """
        # TODO: Insert code here to write raw phase data to the SLM
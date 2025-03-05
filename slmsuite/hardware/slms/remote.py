"""
Template for writing a subclass for SLM hardware control in :mod:`slmsuite`.
Outlines which SLM superclass functions must be implemented.
"""
from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware.remote import Client, DEFAULT_HOST, DEFAULT_PORT, DEFAULT_TIMEOUT

class RemoteSLM(SLM, Client):
    """
    TODO
    """

    def __init__(
        self,
        name: str,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
        wav_um: float = None,
        settle_time_s: float = None,
    ):
        r"""
        Initialize SLM and attributes.

        Parameters
        ----------
        bitdepth : int
            Depth of SLM pixel well in bits. Defaults to 10.
        wav_um : float
            Wavelength of operation in microns. Defaults to 1 um.
        pitch_um : (float, float)
            Pixel pitch in microns. Defaults to 8 micron square pixels.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.

        Note
        ~~~~
        These arguments, which ultimately are used to instantiate the :class:`.SLM` superclass,
        may be more accurately filled by calling the SLM's SDK functions.
        See the other implemented SLM subclasses for examples.
        """
        # Connect to the server.
        super(Client, self).__init__(name, host, port, timeout)

        # Get information about the SLM from the server.
        pickled = self.com(command="pickle")

        # Instantiate the superclass
        super(SLM, self).__init__(
            (pickled["shape"][1], pickled["shape"][0]),
            bitdepth=pickled["bitdepth"],
            name=self.name,
            wav_um=pickled["wav_um"] if wav_um is None else wav_um,
            wav_design_um=pickled["wav_design_um"],
            pitch_um=pickled["pitch_um"],
            settle_time_s=pickled["settle_time_s"] if settle_time_s is None else settle_time_s,
        )

    def _set_phase_hw(self, phase):
        """
        Low-level hardware interface to set_phase ``phase`` data onto the SLM.
        When the user calls the :meth:`.SLM.write` method of
        :class:`.SLM`, ``phase`` is error checked before calling
        :meth:`_set_phase_hw()`. See :meth:`.SLM._set_phase_hw` for further detail.
        """
        # TODO: Insert code here to write raw phase data to the SLM.
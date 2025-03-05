import numpy as np
import socket, sys, json, time
import urllib.parse as urllib
from datetime import date, datetime, timedelta
import traceback
from typing import Any
import zlib

from slmsuite.hardware import _Picklable

DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 5025             # Commonly used for instrument control.
DEFAULT_TIMEOUT = 5
SERVER_WAIT_TIMEOUT = 0.5

delim = "\n"

def _recurse_decompress(msg):
    """
    Recursively decompresses the result of json serialization.
    """
    if isinstance(msg[k], dict):
        if "__zlib__" in msg[k] and len(msg[k]) == 1:
            msg[k] = np.frombuffer(zlib.decompress(msg[k]["__zlib__"]))
        else:
            for k in msg:
                _recurse_decompress(msg[k])

def _recv(sock, timeout):
    recv_buffer = 4096
    buffer = ""
    t = time.time()

    # Pull data into the buffer until we hit timeout or deliminator.
    while time.time() - t < timeout:
        data = sock.recv(recv_buffer).decode()

        buffer += data
        if data[-1] == delim:
            msg = json.loads(urllib.unquote_plus(buffer[0:-len(delim)]))

            _recurse_decompress(msg)

            return msg

    # Failed timeout returns empty.
    return {}

# https://codetinkering.com/numpy-encoder-json/
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.floating): #, np.complexfloating
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            # If the array is above a certain size, compress it. Otherwise return it as a list.
            if obj.size < 500:
                return obj.tolist()
            else:
                return {"__zlib__" : zlib.compress(obj.tobytes())}
        if isinstance(obj, np.string_):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return str(obj)
        if isinstance(obj, np.dtype):
            return {"__dtype__" : str(obj)}
        return super(NpEncoder, self).default(obj)


class Server:
    """
    Server for handling client commands and interfacing with hardware.
    """

    def __init__(
            self,
            hardware: dict,
            port: int = DEFAULT_PORT,
            timeout: float = SERVER_WAIT_TIMEOUT,
            allowlist: list = None,
        ):
        self.hardware = hardware
        self.port = port
        self.timeout = timeout
        self.allowlist = allowlist

    def listen(self, verbose=True):
        """
        Listen for client commands, then process them once they are given.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(self.timeout)
        sock.bind(("", self.port))
        sock.listen(5)

        i = 0

        while True:
            i += 1
            try:
                # Wait for connection with fancy print.
                if verbose:
                    print("Waiting for connection" + ("." * (1 + (i % 3))) + "     ", end="\r")
                connection, client_addr = sock.accept()
                if verbose:
                    print("                              ", end="\r")

                # Check if the client is allowed to connect.
                if self.allowlist is not None:
                    if client_addr[0] not in self.allowlist:
                        connection.close()
                        continue

                # Receive, handle, and reply to message.
                message = _recv(connection, self.timeout)
                result = self.handle(message, client_addr, verbose)
                connection.sendall((urllib.quote_plus(json.dumps(result, cls=NpEncoder)) + delim).encode())

                connection.close()
            except IOError as e:
                pass
            except KeyboardInterrupt:
                if verbose:
                    print("Closing server! Goodbye!")
                try:
                    connection.close()
                except:
                    pass
                sock.close()
                break
            except Exception as e:
                if verbose:
                    print(traceback.format_exc())
                try:
                    connection.close()
                except:
                    pass
                sock.close()
                raise e

    def handle(self, message, client_addr=None, verbose=False) -> tuple[bool, Any]:
        """
        Handle a message from a client.
        """
        try:
            command = message.pop("command")
            name = message.pop("name")
            args = message.pop("args")
            kwargs = message.pop("kwargs")

            instrument = f"hw['{name}'].{command}"

            if verbose: print(f"Received call to {instrument} from client {client_addr}")

            # Initial parse of command.
            if command is None:
                return False, "No command provided."
            elif command == "ping":
                return True, list(self.hardware.keys())

            # Make sure that the hardware exists.
            if not name in self.hardware:
                return False, f"Did not recognize hardware '{name}'."

            if hasattr(self.hardware[name], command):
                if callable(getattr(self.hardware[name], command)):
                    return True, getattr(self.hardware[name], command)(*args, **kwargs)
                # elif "set" in message:
                #     setattr(self.hardware[name], command, message["set"])
                #     return True, True
                else:
                    return False, f"{instrument} is not callable."
            else:
                return False, f"{instrument} not present."
        except:
            return False, traceback.format_exc()


class _Client(_Picklable):
    """
    Client for interfacing with a slmsuite server.
    """

    def __init__(
            self,
            name: str,
            host: str = DEFAULT_HOST,
            port: int = DEFAULT_PORT,
            timeout: float = DEFAULT_TIMEOUT,
        ):
        self.name = name
        self.host = host
        self.port = port
        self.timeout = timeout

        try:
            pickled = self.com(command="ping")
        except:
            raise RuntimeError(
                f"A slmsuite server is not active at {self.host}:{self.port}."
            )

        try:
            pickled = self.com(command="pickle")
        except:
            raise RuntimeError(
                f"Could not connect to {self.name} at {self.host}:{self.port}."
            )

        super().__init__(**pickled)

    def com(
        self,
        command: str = "ping",
        args: list = [],
        kwargs: dict = {},
    ):
        return _Client._com(self.host, self.port, self.timeout, command, args, kwargs)

    @staticmethod
    def _com(
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
        command: str = "ping",
        args: list = [],
        kwargs: dict = {},
    ):
        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))

        # Send the message.
        sock.sendall((
            urllib.quote_plus(
                json.dumps(
                    {
                        "command": command,
                        "args": args,
                        "kwargs": kwargs
                    },
                    cls=NpEncoder
                )
            ) + delim
        ).encode())

        # Wait for a reply.
        try:
            success, reply = _recv(sock, timeout)
            if success == False:
                raise RuntimeError(reply)
        except Exception as e:
            sock.close()

            raise e

        # Always close.
        sock.close()

        return reply

    @staticmethod
    def info(
        verbose: bool = True,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Looks to see if a slmsuite server is active at the given host and port.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Returns
        --------
        list of str
            List of hardware at the server. Returns empty list if no server is found.
        """
        try:
            hardware = _Client._com(host, port, timeout, command="ping")
        except TimeoutError:
            raise TimeoutError("Did not find a server at {host}:{port}.")
        except Exception as e:
            raise e

        if verbose:
            if len(hardware) == 0:
                print("Server found at {host}:{port} with no hardware.")
            else:
                print("Server found at {host}:{port} with hardware:")
                print("\n    ".join(hardware))

        return hardware

    def pickle(self, attributes=False, metadata=True):
        """
        Returns a dictionary containing selected attributes of this class.

        Parameters
        ----------
        attributes : bool OR list of str
            If ``False``, pickles only baseline attributes, usually single floats.
            If ``True``, also pickles 'heavy' attributes such as large images and calibrations.
            If ``list of str``, pickles the keys in the given list.
            By default, the chosen attributes should be things that can be written to
            .h5 files: scalars and lists of scalars.
        metadata : bool
            If ``True``, package the dictionary as the
            ``"__meta__"`` value of a superdictionary which also contains:
            ``"__version__"``, the current slmsuite version,
            ``"__time__"``, the time formatted as a date string, and
            ``"__timestamp__"``, the time formatting as a floating point timestamp.
            This information is used as standard metadata for calibrations and saving.
        """
        t = time.perf_counter()
        response = self.com(command="pickle", kwargs=dict(attributes=False, metadata=True))
        t = time.perf_counter() - t

        data = response.pop("__meta__")
        server_metadata = response
        server_metadata.update(
            {
                "__host__" : self.host,
                "__port__" : self.port,
                "__timeout__" : self.timeout,
                "__latency__" : t
            }
        )

        pickled = {
            "__server__" : server_metadata
        }
        pickled.update(response["__meta__"])

        if metadata:
            t = datetime.datetime.now()
            return {
                "__version__" : __version__,
                "__time__" : str(t),
                "__timestamp__" : t.timestamp(),
                "__meta__" : pickled
            }
        else:
            return pickled

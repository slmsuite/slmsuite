import numpy as np
import socket, sys, json, time
import urllib.parse as urllib
from datetime import date, datetime, timedelta
import traceback
from typing import Any, List, Tuple
import zlib
import base64

from slmsuite.hardware import _Picklable
from slmsuite import __version__

DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 5025             # Commonly used for instrument control.
DEFAULT_TIMEOUT = 5
SERVER_WAIT_TIMEOUT = 0.5

delim = "\n"

def _recurse_decompress(msg):
    """
    Recursively decompresses the result of json serialization.
    """
    if isinstance(msg, dict):
        for k in msg:
            if isinstance(msg[k], dict):
                if "__zlib__" in msg[k] and len(msg[k]) == 1:
                    msg[k] = np.frombuffer(zlib.decompress(msg[k]["__zlib__"]))
                elif "__dtype__" in msg[k] and len(msg[k]) == 1:
                    msg[k] = np.dtype(msg[k]["__dtype__"])
                else:
                    _recurse_decompress(msg[k])

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
            # If the array is above a certain arbitrary size, compress it. Otherwise return it as a list.
            if obj.size > 100:
                return {"__zlib__" : zlib.compress(obj.tobytes())}
            else:
                return obj.tolist()
        if isinstance(obj, np.string_):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return str(obj)
        if isinstance(obj, np.dtype):
            return {"__dtype__" : obj.descr}
        return super(NpEncoder, self).default(obj)

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

            print(msg)

            _recurse_decompress(msg)

            return msg

    # Failed timeout returns empty.
    return {}

class Server:
    """
    Server for handling client commands and interfacing with hardware.
    """

    def __init__(
            self,
            hardware: List[object],
            port: int = DEFAULT_PORT,
            timeout: float = SERVER_WAIT_TIMEOUT,
            allowlist: List[str] = None,
        ):
        """
        TODO
        """
        # Identity.
        self.hardware = {
            hw.name : hw for hw in hardware
        }
        self.port = port
        self.timeout = timeout

        # Only allow clients in the allowlist to connect.
        self.allowlist = allowlist

        # Only allowed commands for SLMs and Cameras, alongside "ping".
        self.allowcommands = [
            "pickle",
            "_set_phase_hw",
            "_set_exposure_hw",
            "_get_exposure_hw",
            "_get_image_hw",
            "_get_images_hw",
        ]

    def listen(self, verbose: bool = True):
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
                if (self.allowlist is not None) and (client_addr[0] not in self.allowlist):
                    result = False, f"Client {client_addr} not in allowlist."
                else:
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

    def handle(
        self,
        message : str,
        client_addr: str = None,
        verbose: bool = False
    ) -> Tuple[bool, Any]:
        """
        Handle a message from a client.
        """
        try:
            name = message.pop("name", None)
            command = message.pop("command", None)
            args = message.pop("args", [])
            kwargs = message.pop("kwargs", dict())

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

            if command in self.allowcommands and hasattr(self.hardware[name], command):
                attribute = getattr(self.hardware[name], command)
                if callable(attribute):
                    return True, attribute(*args, **kwargs)
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
        """

        """
        self.name = name
        self.host = host
        self.port = port
        self.timeout = timeout

        try:
            self.com(command="ping")
        except TimeoutError:
            raise TimeoutError(
                f"An slmsuite server is not active at {self.host}:{self.port}."
            )
        except Exception as e:
            raise e

        try:
            t = time.perf_counter()
            pickled = self.com(
                command="pickle",
                kwargs=dict(attributes=False, metadata=True)
            )
            t = time.perf_counter() - t
        except:
            raise RuntimeError(
                f"Could not connect to '{self.name}' at {self.host}:{self.port}."
            )

        print(pickled)

        self.latency_s = t
        self.server_attributes = pickled

    def com(
        self,
        command: str = "ping",
        args: list = [],
        kwargs: dict = {},
    ):
        return _Client._com(self.name, self.host, self.port, self.timeout, command, args, kwargs)

    @staticmethod
    def _com(
        name: str,
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
                        "name": name,
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
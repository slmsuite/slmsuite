"""
Utilities for interfacing with files.
This includes helper functions for naming directories without conflicts, and convenience
wrappers for file writing. :mod:`slmsuite` uses the
`HDF5 filetype <https://hdfgroup.org/solutions/hdf5>`_
(.h5) by default, as it is fast, compact, and
`widely supported by programming languages for scientific computing
<https://en.wikipedia.org/wiki/Hierarchical_Data_Format#Interfaces>`_.
This uses the :mod:`h5py` `module <https://h5py.org>`_.
"""

import os
import re

import h5py
import numpy as np


def _max_numeric_id(path, name, extension=None, kind="file", digit_count=5):
    """
    Obtains the maximum numeric identifier ``id`` for the
    directory or file like ``path/name_id.extension``.

    Parameters
    ----------
    path
        See :meth:`generate_path`.
    name
        See :meth:`generate_path`.
    extension
        See :meth:`generate_path`.
    kind
        See :meth:`generate_path`.
    digit_count
        See :meth:`generate_path`.

    Returns
    -------
    max_numeric_id : int
         The maximum numeric identifier of the specified object or -1 if no object with
         ``name`` in ``path`` could be found.
    """
    # Search all objects in path for conflicts.
    conflict_regex = "{}_{}{}{}".format(name, r"\d{", digit_count, r"}")
    if extension is not None and kind == "file":
        conflict_regex = "{}.{}".format(conflict_regex, extension)
    max_numeric_id = -1
    for name_ in os.listdir(path):
        # Check current object for conflict.
        conflict = re.search(conflict_regex, name_) is not None
        # Get numeric identifier from conflicting object.
        if conflict:
            suffix = name_.split("{}_".format(name))[1]
            numeric_id = int(suffix[:digit_count])
            max_numeric_id = max(numeric_id, max_numeric_id)

    return max_numeric_id


def generate_path(path, name, extension=None, kind="file", digit_count=5, path_count=1):
    """
    Generate a file path like ``path/name_id.extension``
    or a directory path like ``path/name_id``
    where ``id`` is a unique numeric identifier.

    Parameters
    ----------
    path : str
        Top level directory to create the object in.
        If ``path`` does not exist, it and nonexistent
        parent directories will be created.
    name : str
        Identifier for the object. This should not contain underscores.
    extension : str or None
        The extension to append to the file name, not including the ``.`` separator.
        If ``None``, no extension will be added.
    kind : {"file", "dir"}
        String that identifies what type of object to create.
        If ``kind="dir"`` the directory will be created.
    digit_count : int
        The number of digits to use in the numeric identifier.
    path_count : int
        The number of paths to create. Currently only applicable to file creation.

    Returns
    -------
    file_path : str or list of str
        The full path or paths requested.

    Notes
    -----
    This function is not thread safe.
    """
    path = os.path.abspath(path)
    # Ensure path exists.
    os.makedirs(path, exist_ok=True)

    # Create the name
    max_numeric_id = _max_numeric_id(
        path, name, extension=extension, kind=kind, digit_count=digit_count
    )
    name_format = "{{}}_{{:0{}d}}".format(digit_count)
    name_augmented = name_format.format(name, max_numeric_id + 1)
    if extension is not None and kind == "file":
        name_augmented = "{}.{}".format(name_augmented, extension)
    name_augmented = os.path.join(path, name_augmented)

    # If it's a directory, create it.
    if kind == "dir":
        os.makedirs(name_augmented)

    ret = None
    if path_count == 1:
        ret = name_augmented
    else:
        ret = list()
        for path_idx in range(path_count):
            name_augmented = name_format.format(name, max_numeric_id + 1 + path_idx)
            if extension is not None and kind == "file":
                name_augmented = "{}.{}".format(name_augmented, extension)
            name_augmented = os.path.join(path, name_augmented)
            ret.append(name_augmented)
        # ENDFOR
    # ENDIF

    return ret


def latest_path(path, name, extension=None, kind="file", digit_count=5):
    """
    Obtains the path for the file or directory in ``path`` like ``path/name_id``
    where ``id`` is the greatest identifier in ``path`` for the given ``name``.

    Parameters
    ----------
    path
        See :meth:`generate_path`.
    name
        See :meth:`generate_path`.
    extension
        See :meth:`generate_path`.
    kind
        See :meth:`generate_path`.
    digit_count
        See :meth:`generate_path`.

    Returns
    -------
    file_path : str or None
        The path requested. ``None`` if no file could be found.
    """
    ret = None
    max_numeric_id = _max_numeric_id(
        path, name, extension=extension, kind=kind, digit_count=digit_count
    )
    if max_numeric_id != -1:
        name_format = "{{}}_{{:0{}d}}".format(digit_count)
        name_augmented = name_format.format(name, max_numeric_id)
        if extension is not None and kind == "file":
            name_augmented = "{}.{}".format(name_augmented, extension)
        ret = os.path.join(path, name_augmented)

    return ret


def read_h5(file_path, decode_bytes=True):
    """
    Read data from an h5 file into a dictionary.
    In the case of more complicated h5 hierarchy, a dictionary of dictionaries is returned.

    Parameters
    ----------
    file_path : str
        Full path to the file to read the data from.
    decode_bytes : bool
        Whether or not objects with type `bytes` should be decoded.
        By default HDF5 writes strings as bytes objects; this functionality
        will make strings read back from the file `str` type.

    Returns
    -------
    data : dict
        Dictionary of the data stored in the file.
    """
    def recurse(group):
        data = {}

        for key in group.keys():
            if isinstance(group[key], h5py.Group):
                data[key] = recurse(group[key])
            else:
                data_ = group[key][()]
                if decode_bytes:
                    if isinstance(data_, bytes):
                        data_ = bytes.decode(data_)
                    elif isinstance(data_, np.ndarray) and isinstance(data_[0], bytes):
                        data_ = np.vectorize(bytes.decode)(data_)
                data[key] = data_

        return data

    with h5py.File(file_path, "r") as file_:
        data = recurse(file_)

    return data


def write_h5(file_path, data, mode="w"):
    """
    Write data in a dictionary to an `h5 file
    <https://docs.h5py.org/en/stable/high/file.html#opening-creating-files>`_.

    Parameters
    ----------
    file_path : str
        Full path to the file to save the data in.
    data : dict
        Dictionary of data to save in the file.
    mode : str
        The mode to open the file with.
    """
    def recurse(group, data):
        for key in data.keys():
            if isinstance(data[key], dict):
                new_group = group.create_group(key)
                recurse(new_group, data[key])
            else:
                group[key] = data[key]

    with h5py.File(file_path, mode) as file_:
        recurse(file_, data)

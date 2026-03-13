"""
Unit tests for slmsuite.holography.analysis.files module.
"""
import pytest
import tempfile
import os
import h5py
import numpy as np
from unittest.mock import patch, mock_open

from slmsuite.holography.analysis.files import (
    _max_numeric_id, generate_path, latest_path,
    load_h5, save_h5, read_h5, write_h5, _load_image
)


def _touch(path, content="test"):
    """Helper to create a file with minimal content."""
    with open(path, "w") as f:
        f.write(content)


def test_max_numeric_id(subtests):
    """Test _max_numeric_id for files, directories, and edge cases."""
    with subtests.test("no files returns -1"):
        with tempfile.TemporaryDirectory() as d:
            assert _max_numeric_id(d, "test", "txt", "file", 5) == -1

    with subtests.test("finds highest among files"):
        with tempfile.TemporaryDirectory() as d:
            for name in ["test_00001.txt", "test_00003.txt", "test_00002.txt"]:
                _touch(os.path.join(d, name))
            assert _max_numeric_id(d, "test", "txt", "file", 5) == 3

    with subtests.test("finds highest among directories"):
        with tempfile.TemporaryDirectory() as d:
            for name in ["test_00001", "test_00005", "test_00003"]:
                os.makedirs(os.path.join(d, name))
            assert _max_numeric_id(d, "test", None, "dir", 5) == 5

    with subtests.test("ignores non-matching files"):
        with tempfile.TemporaryDirectory() as d:
            for name in ["test_00001.txt", "other_00005.txt", "test_00002.txt", "test.txt"]:
                _touch(os.path.join(d, name))
            assert _max_numeric_id(d, "test", "txt", "file", 5) == 2


def test_generate_path(subtests):
    """Test generate_path for single/multiple paths, directories, and options."""
    with subtests.test("single file path"):
        with tempfile.TemporaryDirectory() as d:
            result = generate_path(d, "test", "txt", "file", 5, 1)
            assert result == os.path.join(d, "test_00000.txt")

    with subtests.test("multiple file paths"):
        with tempfile.TemporaryDirectory() as d:
            result = generate_path(d, "test", "txt", "file", 5, 3)
            assert isinstance(result, list)
            assert len(result) == 3
            assert result[0] == os.path.join(d, "test_00000.txt")
            assert result[1] == os.path.join(d, "test_00001.txt")
            assert result[2] == os.path.join(d, "test_00002.txt")

    with subtests.test("directory path is created"):
        with tempfile.TemporaryDirectory() as d:
            result = generate_path(d, "test", None, "dir", 5, 1)
            assert result == os.path.join(d, "test_00000")
            assert os.path.isdir(result)

    with subtests.test("increments past existing files"):
        with tempfile.TemporaryDirectory() as d:
            for name in ["test_00000.txt", "test_00001.txt"]:
                _touch(os.path.join(d, name))
            result = generate_path(d, "test", "txt", "file", 5, 1)
            assert result == os.path.join(d, "test_00002.txt")

    with subtests.test("creates missing parent directories"):
        with tempfile.TemporaryDirectory() as d:
            nested = os.path.join(d, "nested", "deep")
            result = generate_path(nested, "test", "txt", "file", 5, 1)
            assert os.path.exists(nested)
            assert result == os.path.join(nested, "test_00000.txt")

    with subtests.test("no extension"):
        with tempfile.TemporaryDirectory() as d:
            result = generate_path(d, "test", None, "file", 5, 1)
            assert result == os.path.join(d, "test_00000")

    with subtests.test("custom digit count"):
        with tempfile.TemporaryDirectory() as d:
            result = generate_path(d, "test", "txt", "file", 3, 1)
            assert result == os.path.join(d, "test_000.txt")


def test_latest_path(subtests):
    """Test latest_path for files and directories."""
    with subtests.test("no files returns None"):
        with tempfile.TemporaryDirectory() as d:
            assert latest_path(d, "test", "txt", "file", 5) is None

    with subtests.test("returns path with highest id"):
        with tempfile.TemporaryDirectory() as d:
            for name in ["test_00001.txt", "test_00003.txt", "test_00002.txt"]:
                _touch(os.path.join(d, name))
            result = latest_path(d, "test", "txt", "file", 5)
            assert result == os.path.join(d, "test_00003.txt")

    with subtests.test("works with directories"):
        with tempfile.TemporaryDirectory() as d:
            for name in ["test_00001", "test_00005", "test_00003"]:
                os.makedirs(os.path.join(d, name))
            result = latest_path(d, "test", None, "dir", 5)
            assert result == os.path.join(d, "test_00005")


def test_save_and_load_h5(subtests):
    """Test HDF5 save/load roundtrip for various data types and options."""
    with subtests.test("simple data types"):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            try:
                data = {
                    "integer": 42,
                    "float": 3.14,
                    "string": "hello",
                    "array": np.array([1, 2, 3, 4, 5]),
                    "none_value": None,
                }
                save_h5(tmp.name, data)
                loaded = load_h5(tmp.name)

                assert loaded["integer"] == 42
                assert loaded["float"] == pytest.approx(3.14)
                assert loaded["string"] == "hello"
                np.testing.assert_array_equal(loaded["array"], [1, 2, 3, 4, 5])
                assert loaded["none_value"] == False  # None becomes False
            finally:
                os.unlink(tmp.name)

    with subtests.test("nested dictionaries"):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            try:
                data = {
                    "level1": {
                        "level2": {
                            "value": 123,
                            "array": np.array([[1, 2], [3, 4]]),
                        },
                        "simple": "test",
                    },
                    "top_level": 456,
                }
                save_h5(tmp.name, data)
                loaded = load_h5(tmp.name)

                assert loaded["level1"]["level2"]["value"] == 123
                np.testing.assert_array_equal(
                    loaded["level1"]["level2"]["array"], [[1, 2], [3, 4]]
                )
                assert loaded["level1"]["simple"] == "test"
                assert loaded["top_level"] == 456
            finally:
                os.unlink(tmp.name)

    with subtests.test("string arrays"):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            try:
                data = {
                    "string_array": np.array(["hello", "world", "test"]),
                    "single_string": "test_string",
                }
                save_h5(tmp.name, data)
                loaded = load_h5(tmp.name)

                np.testing.assert_array_equal(
                    loaded["string_array"], ["hello", "world", "test"]
                )
                assert loaded["single_string"] == "test_string"
            finally:
                os.unlink(tmp.name)

    with subtests.test("staggered arrays raise ValueError"):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            try:
                data = {"staggered": [[1, 2], [3, 4, 5]]}
                with pytest.raises(ValueError, match="staggered arrays"):
                    save_h5(tmp.name, data)
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    with subtests.test("decode_bytes=True vs False"):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            try:
                with h5py.File(tmp.name, "w") as f:
                    f["string_data"] = b"hello"
                    f["byte_array"] = np.array([b"hello", b"world"])

                loaded = load_h5(tmp.name, decode_bytes=True)
                assert loaded["string_data"] == "hello"
                np.testing.assert_array_equal(
                    loaded["byte_array"], ["hello", "world"]
                )

                loaded = load_h5(tmp.name, decode_bytes=False)
                assert loaded["string_data"] == b"hello"
                assert isinstance(loaded["byte_array"][0], bytes)
            finally:
                os.unlink(tmp.name)

    with subtests.test("backwards-compat aliases read_h5/write_h5"):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            try:
                write_h5(tmp.name, {"test": 123})
                loaded = read_h5(tmp.name)
                assert loaded["test"] == 123
            finally:
                os.unlink(tmp.name)

    with subtests.test("mode parameter"):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            try:
                save_h5(tmp.name, {"first": 123}, mode="w")
                save_h5(tmp.name, {"second": 456}, mode="w")
                loaded = load_h5(tmp.name)
                assert "second" in loaded
            finally:
                os.unlink(tmp.name)


def test_load_image(subtests):
    """Test _load_image for error handling, basic loading, inversion, and rotation."""
    with subtests.test("file not found raises ValueError"):
        with pytest.raises(ValueError, match="Image not found"):
            _load_image("nonexistent_file.png", (100, 100))

    with subtests.test("basic grayscale load"):
        mock_image = np.ones((100, 100), dtype=np.uint8) * 50
        with patch("cv2.imread", return_value=mock_image) as mock_imread:
            result = _load_image("test.png", (100, 100))
            assert result.shape == (100, 100)
            mock_imread.assert_called_once_with("test.png", 0)

    with subtests.test("bright image is inverted"):
        mock_image = np.ones((100, 100), dtype=np.uint8) * 200
        with patch("cv2.imread", return_value=mock_image):
            with patch("cv2.bitwise_not") as mock_invert:
                mock_invert.return_value = np.ones((100, 100), dtype=np.uint8) * 55
                _load_image("test.png", (100, 100))
                assert mock_invert.call_count >= 1

    with subtests.test("rotation applied"):
        mock_image = np.ones((100, 100), dtype=np.uint8) * 50
        with patch("cv2.imread", return_value=mock_image):
            with patch("scipy.ndimage.rotate", return_value=mock_image) as mock_rot:
                _load_image("test.png", (100, 100), angle=45)
                mock_rot.assert_called_once_with(mock_image, 45)
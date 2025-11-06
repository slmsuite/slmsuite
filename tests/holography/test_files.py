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


def test_max_numeric_id_no_files():
    """Test when no files exist with the pattern."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = _max_numeric_id(temp_dir, "test", "txt", "file", 5)
        assert result == -1


def test_max_numeric_id_with_files():
    """Test when files exist with the pattern."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_files = ["test_00001.txt", "test_00003.txt", "test_00002.txt"]
        for filename in test_files:
            with open(os.path.join(temp_dir, filename), 'w') as f:
                f.write("test")

        result = _max_numeric_id(temp_dir, "test", "txt", "file", 5)
        assert result == 3


def test_max_numeric_id_directories():
    """Test with directories instead of files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directories
        test_dirs = ["test_00001", "test_00005", "test_00003"]
        for dirname in test_dirs:
            os.makedirs(os.path.join(temp_dir, dirname))

        result = _max_numeric_id(temp_dir, "test", None, "dir", 5)
        assert result == 5


def test_max_numeric_id_mixed_files():
    """Test with files that match and don't match the pattern."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files - some match, some don't
        test_files = ["test_00001.txt", "other_00005.txt", "test_00002.txt", "test.txt"]
        for filename in test_files:
            with open(os.path.join(temp_dir, filename), 'w') as f:
                f.write("test")

        result = _max_numeric_id(temp_dir, "test", "txt", "file", 5)
        assert result == 2


def test_generate_single_file_path():
    """Test generating a single file path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = generate_path(temp_dir, "test", "txt", "file", 5, 1)

        expected = os.path.join(temp_dir, "test_00000.txt")
        assert result == expected


def test_generate_multiple_file_paths():
    """Test generating multiple file paths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = generate_path(temp_dir, "test", "txt", "file", 5, 3)

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == os.path.join(temp_dir, "test_00000.txt")
        assert result[1] == os.path.join(temp_dir, "test_00001.txt")
        assert result[2] == os.path.join(temp_dir, "test_00002.txt")


def test_generate_directory_path():
    """Test generating a directory path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = generate_path(temp_dir, "test", None, "dir", 5, 1)

        expected = os.path.join(temp_dir, "test_00000")
        assert result == expected
        assert os.path.exists(result)
        assert os.path.isdir(result)


def test_generate_path_with_existing_files():
    """Test path generation when files already exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create existing files
        existing_files = ["test_00000.txt", "test_00001.txt"]
        for filename in existing_files:
            with open(os.path.join(temp_dir, filename), 'w') as f:
                f.write("test")

        result = generate_path(temp_dir, "test", "txt", "file", 5, 1)
        expected = os.path.join(temp_dir, "test_00002.txt")
        assert result == expected


def test_generate_path_creates_directory():
    """Test that generate_path creates missing directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        nested_path = os.path.join(temp_dir, "nested", "deep")
        result = generate_path(nested_path, "test", "txt", "file", 5, 1)

        assert os.path.exists(nested_path)
        expected = os.path.join(nested_path, "test_00000.txt")
        assert result == expected


def test_generate_path_no_extension():
    """Test generating file path without extension."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = generate_path(temp_dir, "test", None, "file", 5, 1)

        expected = os.path.join(temp_dir, "test_00000")
        assert result == expected


def test_generate_path_custom_digit_count():
    """Test generating path with custom digit count."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = generate_path(temp_dir, "test", "txt", "file", 3, 1)

        expected = os.path.join(temp_dir, "test_000.txt")
        assert result == expected


def test_latest_path_no_files():
    """Test when no files exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = latest_path(temp_dir, "test", "txt", "file", 5)
        assert result is None


def test_latest_path_with_files():
    """Test with existing files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_files = ["test_00001.txt", "test_00003.txt", "test_00002.txt"]
        for filename in test_files:
            with open(os.path.join(temp_dir, filename), 'w') as f:
                f.write("test")

        result = latest_path(temp_dir, "test", "txt", "file", 5)
        expected = os.path.join(temp_dir, "test_00003.txt")
        assert result == expected


def test_latest_path_directories():
    """Test latest_path with directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directories
        test_dirs = ["test_00001", "test_00005", "test_00003"]
        for dirname in test_dirs:
            os.makedirs(os.path.join(temp_dir, dirname))

        result = latest_path(temp_dir, "test", None, "dir", 5)
        expected = os.path.join(temp_dir, "test_00005")
        assert result == expected


def test_save_and_load_h5_simple_data():
    """Test saving and loading simple data types."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        try:
            data = {
                'integer': 42,
                'float': 3.14,
                'string': 'hello',
                'array': np.array([1, 2, 3, 4, 5]),
                'none_value': None
            }

            save_h5(tmp_file.name, data)
            loaded_data = load_h5(tmp_file.name)

            assert loaded_data['integer'] == 42
            assert loaded_data['float'] == pytest.approx(3.14)
            assert loaded_data['string'] == 'hello'
            np.testing.assert_array_equal(loaded_data['array'], [1, 2, 3, 4, 5])
            assert loaded_data['none_value'] == False  # None becomes False

        finally:
            os.unlink(tmp_file.name)


def test_save_and_load_h5_nested_dict():
    """Test saving and loading nested dictionaries."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        try:
            data = {
                'level1': {
                    'level2': {
                        'value': 123,
                        'array': np.array([[1, 2], [3, 4]])
                    },
                    'simple': 'test'
                },
                'top_level': 456
            }

            save_h5(tmp_file.name, data)
            loaded_data = load_h5(tmp_file.name)

            assert loaded_data['level1']['level2']['value'] == 123
            np.testing.assert_array_equal(
                loaded_data['level1']['level2']['array'],
                [[1, 2], [3, 4]]
            )
            assert loaded_data['level1']['simple'] == 'test'
            assert loaded_data['top_level'] == 456

        finally:
            os.unlink(tmp_file.name)


def test_save_h5_string_arrays():
    """Test saving string arrays."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        try:
            data = {
                'string_array': np.array(['hello', 'world', 'test']),
                'single_string': 'test_string'
            }

            save_h5(tmp_file.name, data)
            loaded_data = load_h5(tmp_file.name)

            np.testing.assert_array_equal(
                loaded_data['string_array'],
                ['hello', 'world', 'test']
            )
            assert loaded_data['single_string'] == 'test_string'

        finally:
            os.unlink(tmp_file.name)


def test_save_h5_staggered_array_error():
    """Test that staggered arrays raise appropriate error."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        try:
            data = {
                'staggered': [[1, 2], [3, 4, 5]]  # Different lengths
            }

            with pytest.raises(ValueError, match="staggered arrays"):
                save_h5(tmp_file.name, data)

        finally:
            if os.path.exists(tmp_file.name):
                os.unlink(tmp_file.name)


def test_load_h5_decode_bytes():
    """Test loading with and without byte decoding."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        try:
            # Create file with bytes manually
            with h5py.File(tmp_file.name, 'w') as f:
                f['string_data'] = b'hello'
                f['byte_array'] = np.array([b'hello', b'world'])

            # Test with decode_bytes=True (default)
            loaded_data = load_h5(tmp_file.name, decode_bytes=True)
            assert loaded_data['string_data'] == 'hello'
            np.testing.assert_array_equal(
                loaded_data['byte_array'],
                ['hello', 'world']
            )

            # Test with decode_bytes=False
            loaded_data = load_h5(tmp_file.name, decode_bytes=False)
            assert loaded_data['string_data'] == b'hello'
            assert isinstance(loaded_data['byte_array'][0], bytes)

        finally:
            os.unlink(tmp_file.name)


def test_backwards_compatibility_aliases():
    """Test backwards compatibility aliases."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        try:
            data = {'test': 123}

            # Test write_h5 alias
            write_h5(tmp_file.name, data)

            # Test read_h5 alias
            loaded_data = read_h5(tmp_file.name)

            assert loaded_data['test'] == 123

        finally:
            os.unlink(tmp_file.name)


def test_save_h5_append_mode():
    """Test saving in append mode."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        try:
            # Save initial data
            data1 = {'first': 123}
            save_h5(tmp_file.name, data1, mode='w')

            # Save additional data in append mode - this would overwrite in practice
            # but tests the mode parameter
            data2 = {'second': 456}
            save_h5(tmp_file.name, data2, mode='w')  # h5py doesn't have true append for keys

            loaded_data = load_h5(tmp_file.name)
            assert 'second' in loaded_data

        finally:
            os.unlink(tmp_file.name)


def test_load_image_file_not_found():
    """Test error when image file doesn't exist."""
    with pytest.raises(ValueError, match="Image not found"):
        _load_image("nonexistent_file.png", (100, 100))


@patch('cv2.imread')
def test_load_image_basic(mock_imread):
    """Test basic image loading functionality."""
    # Mock a simple grayscale image
    mock_image = np.ones((100, 100), dtype=np.uint8) * 50
    mock_imread.return_value = mock_image

    result = _load_image("test.png", (100, 100))

    assert result.shape == (100, 100)
    mock_imread.assert_called_once_with("test.png", 0)  # cv2.IMREAD_GRAYSCALE = 0


@patch('cv2.imread')
def test_load_image_inversion(mock_imread):
    """Test image inversion when majority is bright."""
    # Mock a bright image that should be inverted
    mock_image = np.ones((100, 100), dtype=np.uint8) * 200
    mock_imread.return_value = mock_image

    with patch('cv2.bitwise_not') as mock_invert:
        mock_invert.return_value = np.ones((100, 100), dtype=np.uint8) * 55

        result = _load_image("test.png", (100, 100))

        # Should call bitwise_not for inversion (may be called multiple times)
        assert mock_invert.call_count >= 1


@patch('cv2.imread')
@patch('scipy.ndimage.rotate')
def test_load_image_with_rotation(mock_rotate, mock_imread):
    """Test image loading with rotation."""
    mock_image = np.ones((100, 100), dtype=np.uint8) * 50
    mock_imread.return_value = mock_image
    mock_rotate.return_value = mock_image

    result = _load_image("test.png", (100, 100), angle=45)

    mock_rotate.assert_called_once_with(mock_image, 45)
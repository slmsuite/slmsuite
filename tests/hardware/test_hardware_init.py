"""
Unit tests for slmsuite.hardware.__init__ module.
"""
import pytest
import tempfile
import os
import datetime
from unittest.mock import patch
import warnings

from slmsuite.hardware import _Picklable
from slmsuite import __version__


class TestPicklable:
    """Test the _Picklable class functionality."""

    def setup_method(self):
        """Set up a test Picklable object for each test."""
        class TestPicklableClass(_Picklable):
            _pickle = ['basic_attr', 'name']
            _pickle_data = ['heavy_attr']

            def __init__(self):
                self.basic_attr = 42
                self.heavy_attr = [1, 2, 3, 4, 5]
                self.name = "test_object"
                self.unpickled_attr = "not_pickled"

            def __str__(self):
                return "TestPicklableClass"

        self.test_obj = TestPicklableClass()

    def test_pickle_basic_attributes_only(self):
        """Test pickling with attributes=False (basic only)."""
        result = self.test_obj.pickle(attributes=False, metadata=False)

        assert "__class__" in result
        assert result["__class__"] == "TestPicklableClass"
        assert "basic_attr" in result
        assert result["basic_attr"] == 42
        assert "name" in result
        assert result["name"] == "test_object"
        assert "heavy_attr" not in result
        assert "unpickled_attr" not in result

    def test_pickle_all_attributes(self):
        """Test pickling with attributes=True (all)."""
        result = self.test_obj.pickle(attributes=True, metadata=False)

        assert "__class__" in result
        assert result["__class__"] == "TestPicklableClass"
        assert "basic_attr" in result
        assert result["basic_attr"] == 42
        assert "name" in result
        assert result["name"] == "test_object"
        assert "heavy_attr" in result
        assert result["heavy_attr"] == [1, 2, 3, 4, 5]
        assert "unpickled_attr" not in result

    def test_pickle_custom_attributes(self):
        """Test pickling with custom attribute list."""
        result = self.test_obj.pickle(attributes=['basic_attr'], metadata=False)

        assert "__class__" in result
        assert result["__class__"] == "TestPicklableClass"
        assert "basic_attr" in result
        assert result["basic_attr"] == 42
        assert "name" not in result
        assert "heavy_attr" not in result

    def test_pickle_with_metadata(self):
        """Test pickling with metadata enabled."""
        with patch('datetime.datetime') as mock_datetime:
            mock_now = datetime.datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kw: datetime.datetime(*args, **kw)

            result = self.test_obj.pickle(attributes=False, metadata=True)

            assert "__version__" in result
            assert result["__version__"] == __version__
            assert "__time__" in result
            assert result["__time__"] == str(mock_now)
            assert "__timestamp__" in result
            assert result["__timestamp__"] == mock_now.timestamp()
            assert "__meta__" in result

            meta = result["__meta__"]
            assert "__class__" in meta
            assert "basic_attr" in meta
            assert "name" in meta

    def test_pickle_missing_attribute_warning(self):
        """Test that warning is issued for missing attributes."""
        with pytest.warns(UserWarning, match="Expected attribute 'nonexistent' not present"):
            result = self.test_obj.pickle(attributes=['nonexistent'], metadata=False)

        # Should still contain class info
        assert "__class__" in result

    def test_pickle_recursive_picklable_attributes(self):
        """Test pickling with nested Picklable objects."""
        class NestedPicklable(_Picklable):
            _pickle = ['nested_value']

            def __init__(self):
                self.nested_value = "nested"

            def __str__(self):
                return "NestedPicklable"

        self.test_obj.nested_obj = NestedPicklable()

        result = self.test_obj.pickle(attributes=['nested_obj'], metadata=False)

        assert "nested_obj" in result
        nested_result = result["nested_obj"]
        assert "__class__" in nested_result
        assert nested_result["__class__"] == "NestedPicklable"
        assert "nested_value" in nested_result
        assert nested_result["nested_value"] == "nested"

    def test_save_method(self):
        """Test the save method functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test save with default name
            saved_path = self.test_obj.save(path=temp_dir)

            assert os.path.exists(saved_path)
            assert saved_path.endswith('.h5')
            assert 'test_object-pickle' in saved_path

            # Test save with custom name
            custom_path = self.test_obj.save(path=temp_dir, name="custom_name")

            assert os.path.exists(custom_path)
            assert 'custom_name' in custom_path

    def test_save_method_with_kwargs(self):
        """Test save method passes kwargs to pickle."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save with attributes=True to include heavy data
            saved_path = self.test_obj.save(path=temp_dir, attributes=True)

            assert os.path.exists(saved_path)

            # Load and verify heavy data was saved
            import h5py
            with h5py.File(saved_path, 'r') as f:
                assert '__meta__' in f
                meta_group = f['__meta__']
                assert 'heavy_attr' in meta_group

    def test_save_without_name_attribute(self):
        """Test save method when object has no name attribute."""
        class NoNamePicklable(_Picklable):
            _pickle = ['value']

            def __init__(self):
                self.value = 123

            def __str__(self):
                return "NoNamePicklable"

        obj_without_name = NoNamePicklable()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Should raise AttributeError when trying to access name
            with pytest.raises(AttributeError):
                obj_without_name.save(path=temp_dir)

    def test_pickle_edge_cases(self):
        """Test various edge cases for pickle method."""
        # Test with empty _pickle lists
        class EmptyPicklable(_Picklable):
            _pickle = []
            _pickle_data = []

            def __init__(self):
                self.some_attr = "value"

            def __str__(self):
                return "EmptyPicklable"

        empty_obj = EmptyPicklable()
        result = empty_obj.pickle(attributes=True, metadata=False)

        assert "__class__" in result
        assert "some_attr" not in result  # Not in _pickle lists

    def test_str_method_integration(self):
        """Test that __str__ method is properly used in pickle."""
        result = self.test_obj.pickle(attributes=False, metadata=False)
        assert result["__class__"] == "TestPicklableClass"
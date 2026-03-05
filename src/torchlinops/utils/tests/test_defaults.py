import pytest

from torchlinops.utils._defaults import default_to, default_to_dict


class TestDefaultTo:
    def test_empty_vals(self):
        result = default_to()
        assert result is None

    def test_single_value(self):
        result = default_to(5)
        assert result == 5

    def test_first_non_none(self):
        result = default_to(None, 5, 10)
        assert result == 10

    def test_all_none(self):
        result = default_to(None, None, None)
        assert result is None

    def test_middle_none(self):
        result = default_to(5, None, 10)
        assert result == 10

    def test_right_to_left(self):
        result = default_to(None, None, 42)
        assert result == 42


class TestDefaultToDict:
    def test_empty_args(self):
        result = default_to_dict()
        assert result == {}

    def test_single_dict(self):
        result = default_to_dict({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_multiple_dicts(self):
        result = default_to_dict({"a": 1}, {"b": 2}, {"c": 3})
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_later_dict_overwrites(self):
        result = default_to_dict({"a": 1}, {"a": 99})
        assert result == {"a": 99}

    def test_none_dicts_ignored(self):
        result = default_to_dict({"a": 1}, None, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_non_dict_raises(self):
        with pytest.raises(ValueError):
            default_to_dict({"a": 1}, "not a dict", {"b": 2})

    def test_multiple_nones(self):
        result = default_to_dict(None, None, None)
        assert result == {}

    def test_none_in_middle(self):
        result = default_to_dict({"a": 1}, None, {"b": 2})
        assert result == {"a": 1, "b": 2}


class TestDefaultToDict:
    def test_empty_args(self):
        result = default_to_dict()
        assert result == {}

    def test_single_dict(self):
        result = default_to_dict({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_multiple_dicts(self):
        result = default_to_dict({"a": 1}, {"b": 2}, {"c": 3})
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_later_dict_overwrites(self):
        result = default_to_dict({"a": 1}, {"a": 99})
        assert result == {"a": 99}

    def test_none_dicts_ignored(self):
        result = default_to_dict({"a": 1}, None, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_non_dict_raises(self):
        with pytest.raises(ValueError):
            default_to_dict({"a": 1}, "not a dict", {"b": 2})

    def test_multiple_nones(self):
        result = default_to_dict(None, None, None)
        assert result == {}

    def test_none_in_middle(self):
        result = default_to_dict({"a": 1}, None, {"b": 2})
        assert result == {"a": 1, "b": 2}

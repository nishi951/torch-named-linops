import pytest

from torchlinops.utils._dispatch import build_signature, check_signature


class TestCheckSignature:
    def test_positional_args_match(self):
        spec = [("arg1", int), ("arg2", str)]
        result = check_signature(spec, 10, "test")
        assert result is not None

    def test_positional_args_mismatch(self):
        spec = [("arg1", int), ("arg2", str)]
        result = check_signature(spec, "test", 10)
        assert result is None

    def test_kwargs_match(self):
        spec = [("arg1", int), ("arg2", str)]
        result = check_signature(spec, arg1=5, arg2="hello")
        assert result is not None

    def test_kwargs_mismatch(self):
        spec = [("arg1", int)]
        result = check_signature(spec, arg1="not an int")
        assert result is None

    def test_kwargs_with_wrong_type(self):
        spec = [("arg1", int), ("arg2", str)]
        result = check_signature(spec, 10, arg2=123)
        assert result is None

    def test_empty_args_match(self):
        spec = []
        result = check_signature(spec)
        assert result is not None


class TestBuildSignature:
    def test_simple_params(self):
        spec = [("arg1", int), ("arg2", str)]
        sig, allow_kwargs = build_signature(spec)
        assert sig is not None
        assert not allow_kwargs

    def test_with_defaults(self):
        spec = [("arg1", int, 10), ("arg2", str, "default")]
        sig, allow_kwargs = build_signature(spec)
        params = list(sig.parameters.values())
        assert params[0].default == 10
        assert params[1].default == "default"

    def test_with_annotations_only(self):
        spec = [("arg1", int), ("arg2", str)]
        sig, allow_kwargs = build_signature(spec)
        params = list(sig.parameters.values())
        assert params[0].annotation == int
        assert params[1].annotation == str

    def test_anonymous_var_keyword(self):
        spec = ["**"]
        sig, allow_kwargs = build_signature(spec)
        params = list(sig.parameters.values())
        assert params[0].kind == 4
        assert allow_kwargs

    def test_invalid_tuple_token_length(self):
        spec = [("arg", int, "default", "extra")]
        with pytest.raises(ValueError):
            build_signature(spec)

    def test_invalid_token(self):
        spec = [123]
        with pytest.raises(ValueError):
            build_signature(spec)

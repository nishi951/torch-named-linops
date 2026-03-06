from torchlinops.utils._dispatch import build_signature, check_signature


def test_check_signature_match():
    result = check_signature([("arg1", int), ("arg2", str)], 10, "test")
    assert result is not None


def test_check_signature_mismatch():
    result = check_signature([("arg1", int), ("arg2", str)], "test", 10)
    assert result is None


def test_check_signature_too_few_args():
    result = check_signature([("arg1", int), ("arg2", str)], 10)
    assert result is None


def test_build_signature_positional_only():
    sig, allow_kwargs = build_signature([("a",), "/"])
    assert not allow_kwargs


def test_build_signature_keyword_only():
    sig, allow_kwargs = build_signature([("a",), "*", ("b",)])
    assert not allow_kwargs


def test_build_signature_var_keyword():
    sig, allow_kwargs = build_signature([("a",), "**kwargs"])
    assert allow_kwargs


def test_build_signature_var_positional():
    sig, allow_kwargs = build_signature(["*args"])
    assert not allow_kwargs


def test_build_signature_default():
    sig, allow_kwargs = build_signature([("a",), ("b", 42)])
    assert not allow_kwargs


def test_build_signature_keyword_only():
    sig, allow_kwargs = build_signature([("a",), "*", ("b",)])
    assert not allow_kwargs


def test_build_signature_var_keyword():
    sig, allow_kwargs = build_signature([("a",), "**kwargs"])
    assert allow_kwargs


def test_build_signature_var_positional():
    sig, allow_kwargs = build_signature(["*args"])
    assert not allow_kwargs


def test_build_signature_default():
    sig, allow_kwargs = build_signature([("a",), ("b", 42)])
    assert not allow_kwargs

import pytest

from torchlinops import config


def test_using_single_key():
    original = config.reduce_identity_in_normal
    with config.using(reduce_identity_in_normal=not original):
        assert config.reduce_identity_in_normal == (not original)
    assert config.reduce_identity_in_normal == original


def test_using_multiple_keys():
    orig_reduce = config.reduce_identity_in_normal
    orig_log = config.log_device_transfers
    with config.using(
        reduce_identity_in_normal=not orig_reduce, log_device_transfers=not orig_log
    ):
        assert config.reduce_identity_in_normal == (not orig_reduce)
        assert config.log_device_transfers == (not orig_log)
    assert config.reduce_identity_in_normal == orig_reduce
    assert config.log_device_transfers == orig_log


def test_using_nested():
    orig = config.reduce_identity_in_normal
    with config.using(reduce_identity_in_normal=False):
        assert config.reduce_identity_in_normal is False
        with config.using(reduce_identity_in_normal=True):
            assert config.reduce_identity_in_normal is True
        assert config.reduce_identity_in_normal is False
    assert config.reduce_identity_in_normal is orig


def test_using_restores_on_exception():
    orig = config.reduce_identity_in_normal
    with pytest.raises(ValueError):
        with config.using(reduce_identity_in_normal=not orig):
            assert config.reduce_identity_in_normal == (not orig)
            raise ValueError("test exception")
    assert config.reduce_identity_in_normal == orig


def test_using_invalid_key_raises():
    with pytest.raises(ValueError) as exc_info:
        config.using(invalid_key=True)
    assert "Invalid config key" in str(exc_info.value)
    assert "invalid_key" in str(exc_info.value)


def test_using_valid_keys():
    config.using(reduce_identity_in_normal=True)
    config.using(cache_adjoint_normal=True)
    config.using(log_device_transfers=True)
    config.using(
        reduce_identity_in_normal=False,
        cache_adjoint_normal=False,
        log_device_transfers=False,
    )

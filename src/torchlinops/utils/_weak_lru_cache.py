import weakref
from functools import lru_cache, update_wrapper

__all__ = ["weak_lru_cache"]


def _make_weak_bound_method(func, wref):
    """Wrap *func* so it dereferences a weakref to its instance on each call.

    Strong references to the instance never enter the cache, so instances
    stay collectible and reference cycles (instance -> cache -> instance)
    are avoided.
    """

    def call(*args, **kwargs):
        instance = wref()
        return func(instance, *args, **kwargs)

    return call


class CacheBox:
    """Callable box holding one instance's lru_cache.

    Stored directly on the owning instance's ``__dict__``. Owning instance
    identity and owning wrapper identity are embedded so that copies
    (shallow or deep) of the instance, or a stale box from a different
    wrapper, can be detected and rebound on first access.
    """

    __slots__ = ("_cache", "_owner_wr", "_wrapper")

    def __init__(self, wrapper=None, owner_wr=None, cache=None):
        self._wrapper = wrapper
        self._owner_wr = owner_wr
        self._cache = cache

    def __call__(self, *args, **kwargs):
        return self._cache(*args, **kwargs)

    def cache_clear(self):
        self._cache.cache_clear()

    def cache_info(self):
        return self._cache.cache_info()

    @property
    def __wrapped__(self):
        return self._cache.__wrapped__

    def __deepcopy__(self, memo):
        # Copies never inherit hot verdicts; they rebind lazily on first access
        return CacheBox()

    def __reduce__(self):
        return _cold_box, ()


def _cold_box():
    """Reconstruct a cache-less box after unpickling (no mutable shared state)."""
    return CacheBox()


class WeakLRUMethodWrapper:
    """A modified version of lru_cache that can be applied to an instance method.

    Unlike ``lru_cache`` on a plain function, each instance gets an independent,
    per-instance cache that is stored on the instance itself. No shared registries
    (WeakKeyDictionary etc.) are maintained, so this decorator is safe to use
    concurrently on multiple instances, including on free-threaded CPython.

    The underlying function is bound to its instance only *weakly*, which means
    a cached method can never keep its instance alive.
    """

    def __init__(self, func, maxsize):
        self.func = func
        self.maxsize = maxsize
        self.__wrapped__ = func
        update_wrapper(self, func)
        # Deterministic name for the instance-level cache attribute. Uniqueness
        # is not load-bearing: the identity check in __get__ detects and rebinds
        # stale boxes even if another wrapper ever shares this name.
        self._cache_attr = f"_{func.__name__}__weaklrucache"

    def __get__(self, instance, owner):
        if instance is None:
            return self  # Called on the class itself

        inst_dict = instance.__dict__
        box = inst_dict.get(self._cache_attr)
        # A box is reusable iff it is ours and it belongs to *this* instance.
        # Copies of the instance (shallow copies share __dict__; deepcopies
        # and unpickled instances receive a cold box) are rebound here.
        if (
            box is None
            or box._wrapper is not self  # box came from some other wrapper
            or box._owner_wr is None  # cold box from a deepcopy/unpickle
            or box._owner_wr() is not instance  # box belongs to the original of a copy
        ):
            wref = weakref.ref(instance)  # strong refs to instance stay out of the box
            shim = _make_weak_bound_method(self.func, wref)
            cache = lru_cache(maxsize=self.maxsize)(shim)
            box = CacheBox(self, wref, cache)
            inst_dict[self._cache_attr] = box
        return box


def weak_lru_cache(maxsize=128):
    """Decorator to cache results of instance methods on the instance itself."""

    def decorator(func):
        return WeakLRUMethodWrapper(func, maxsize)

    return decorator

import copy
import gc
import pickle
import weakref

import pytest

from torchlinops.utils import weak_lru_cache
from torchlinops.utils._weak_lru_cache import CacheBox, WeakLRUMethodWrapper


class Counter:
    count = 0

    def __init__(self, value=0):
        self.value = value

    @weak_lru_cache(maxsize=8)
    def f(self, a, b=2):
        Counter.count += 1
        return a + b + self.value


def cache_attr():
    wrapper = Counter.__dict__["f"]
    assert isinstance(wrapper, WeakLRUMethodWrapper)
    return wrapper._cache_attr


@pytest.fixture(autouse=True)
def reset_counter():
    Counter.count = 0


@pytest.fixture
def counter():
    return Counter()


def test_results_consistent(counter):
    assert counter.f(1) == 3
    assert counter.f(1) == 3
    assert Counter.count == 1
    assert counter.f(1, b=5) == 6
    assert Counter.count == 2
    assert counter.f(1) == 3  # still cached
    assert Counter.count == 2


def test_independent_instance_caches():
    a, b = Counter(value=1), Counter(value=100)
    assert a.f(1) == 4
    assert b.f(1) == 103
    assert Counter.count == 2
    assert a.f(1) == 4  # a's cache is not polluted by b's calls
    assert b.f(1) == 103
    assert Counter.count == 2


def test_lru_eviction(counter):
    for i in range(10):  # more calls than maxsize=8
        counter.f(i)
    assert Counter.count == 10
    # The two oldest entries (0, 1) were evicted
    counter.f(0)
    assert Counter.count == 11
    # ...but recently-used entries are still cached
    counter.f(9)
    assert Counter.count == 11


def test_result_consistent_after_eviction(counter):
    results = {i: counter.f(i) for i in range(9)}  # 9 calls, maxsize 8 -> 0 evicted
    assert Counter.count == 9
    assert counter.f(0) == results[0]  # recompute gives same result
    assert Counter.count == 10


def test_exposes_lru_interface(counter):
    counter.f(1)
    proxy = counter.f
    info = proxy.cache_info()
    assert (info.hits, info.misses) == (0, 1)  # the first call was a miss
    assert counter.f(1) == 3
    assert proxy.cache_info().hits == 1
    proxy.cache_clear()
    assert proxy.cache_info().currsize == 0
    counter.f(1)  # uncached after clear
    assert Counter.count == 2
    assert proxy.cache_info().currsize == 1  # post-clear stats restart at 0


def test_kwargs_normalize_with_positional(counter):
    assert counter.f(1, b=1) == counter.f(1, 1)


def test_instance_is_not_pinned_by_its_cache():
    """The cache must not keep a dead instance alive."""
    c = Counter(value=7)
    c.f(3)
    ref = weakref.ref(c)
    del c
    gc.collect()
    assert ref() is None


def test_class_access_returns_wrapper():
    wrapper = Counter.__dict__["f"]
    assert wrapper.__get__(None, Counter) is wrapper


def test_stale_box_from_other_wrapper_is_rebound():
    """A box from another wrapper sharing the attr name (re-decoration /
    same-named method wrapped twice) must be detected and rebound."""
    from functools import lru_cache

    from torchlinops.utils._weak_lru_cache import WeakLRUMethodWrapper

    real = Counter.__dict__["f"]
    foreign = WeakLRUMethodWrapper(real.func, maxsize=8)  # same _cache_attr name
    assert foreign._cache_attr == real._cache_attr
    c = Counter(value=3)
    # Foreign wrapper grabs the slot first and computes through its shim
    foreign_box = foreign.__get__(c, Counter)
    assert foreign_box(1) == 6
    assert Counter.count == 1
    # Calling through the real wrapper must detect the foreign box and rebind
    assert c.f(1) == 6
    assert c.__dict__[real._cache_attr]._wrapper is real
    assert Counter.count == 2  # fresh computation, foreign verdict discarded


def test_cache_is_stored_on_the_instance(counter):
    """Calling the method writes the cache box onto the instance itself."""
    attr = cache_attr()
    assert attr not in counter.__dict__
    counter.f(1)
    box = counter.__dict__[attr]
    assert isinstance(box, CacheBox)
    # other instances do not get a cache until they call f
    other = Counter()
    assert (attr in other.__dict__) is False


def test_shallow_copy_rebinds_to_the_copy():
    """copy.copy shares instance __dict__; copies must never use the original's per-instance cache."""
    attr = cache_attr()
    a = Counter(value=1)
    assert a.f(1) == 4
    b = copy.copy(a)
    b.value = 100
    assert b.f(1) == 103  # if b reused a's cache it would return 4
    assert a.f(1) == 4  # and a's verdict must stay intact
    assert Counter.count == 2
    assert counter.__dict__ if False else True
    assert a.__dict__[attr] is not b.__dict__[attr]


def test_deepcopy_starts_cold():
    """deepcopy must not carry over cached verdicts, and must bind to the copy."""
    attr = cache_attr()
    a = Counter(value=1)
    assert a.f(1) == 4
    b = copy.deepcopy(a)
    b.value = 50
    assert b.f(1) == 53  # recomputed with b's value, not a's cached verdict
    assert Counter.count == 2
    assert a.__dict__[attr] is not b.__dict__[attr]
    assert a.f(1) == 4  # a's cache untouched
    assert Counter.count == 2


def test_pickle_roundtrip_starts_cold():
    attr = cache_attr()
    a = Counter(value=1)
    assert a.f(1) == 4
    b = pickle.loads(pickle.dumps(a))
    assert Counter.count == 1  # no verdicts resurrected
    b.value = 7
    assert b.f(1) == 10
    assert Counter.count == 2
    assert a.f(1) == 4
    assert Counter.count == 2

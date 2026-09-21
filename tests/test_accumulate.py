"""Unit tests for the streaming/accumulate execution option.

Covers the `accumulate` flag added to `Stack` / `Add` / `Concat` (issue #197:
bound live child outputs by accumulating per batch instead of materializing
all N results before the reduce), its plumbing through
`parallel_execute(threaded=..., num_workers=..., accumulate=...)`, and its
propagation through `create_batched_linop(**options)` (issue #196).

Key contract points:
- accumulate must not change results: concatenation-style reduces must be
  BITWISE equal (chunking preserves element order and copies values);
  summation-style reduces may differ in float association only when
  num_workers batches group terms differently, so they are allclose.
- ragged final batches (len(children) % num_workers != 0) must work.
"""

import pytest
import torch

from torchlinops import Add, BatchSpec, Concat, Dense, Stack, create_batched_linop
from torchlinops.linops.schedule import parallel_execute

COMPLEX64 = torch.complex64


def _scaling_ops(n, size=4, seed=0):
    """n trivial child 'linops' (callables of (x, context)) plus identical inputs."""
    g = torch.Generator().manual_seed(seed)
    scales = torch.randn(n, size, generator=g).to(COMPLEX64)
    ops = [(lambda s: lambda x, context=None: x * s)(scales[i]) for i in range(n)]
    inputs = [torch.ones(size, dtype=COMPLEX64) for _ in range(n)]
    refs = [x * s for x, s in zip(inputs, scales)]
    return ops, inputs, refs


_SUM = lambda ys: sum(ys)


def _sum2(x, y):
    return x + y


_CAT_DIM0 = lambda ys: torch.concatenate(ys, dim=0)


def _cat2(x, y):
    return torch.concatenate((x, y), dim=0)


# ---------------------------------------------------------------- primitive


@pytest.mark.parametrize("n", [1, 2, 3, 5, 6])
@pytest.mark.parametrize("threaded", [False, True])
@pytest.mark.parametrize("num_workers", [None, 1, 2, 3, 10])
def test_parallel_execute_sum_matches_reference(n, threaded, num_workers):
    ops, inputs, refs = _scaling_ops(n)
    ref = _SUM(refs)
    for acc in (False, True):
        out = parallel_execute(
            ops,
            inputs,
            None,
            _SUM,
            threaded=threaded,
            num_workers=num_workers,
            accumulate=acc,
        )
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6), (
            n,
            threaded,
            num_workers,
            acc,
        )


@pytest.mark.parametrize("n", [1, 2, 3, 5, 6])
@pytest.mark.parametrize("threaded", [False, True])
@pytest.mark.parametrize("num_workers", [None, 1, 2, 3, 10])
def test_parallel_execute_concat_is_bitwise_preserved(n, threaded, num_workers):
    """Concatenation reduces copy values in order: chunking must be bitwise-neutral."""
    ops, inputs, refs = _scaling_ops(n)
    ref = _CAT_DIM0(refs)
    for acc in (False, True):
        out = parallel_execute(
            ops,
            inputs,
            None,
            _CAT_DIM0,
            threaded=threaded,
            num_workers=num_workers,
            accumulate=acc,
            accumulate_fn=_cat2 if acc else None,
        )
        assert torch.equal(out, ref), (n, threaded, num_workers, acc)


def test_parallel_execute_default_accumulate_is_on():
    ops, inputs, refs = _scaling_ops(3)
    default = parallel_execute(ops, inputs, None, _SUM, threaded=True, num_workers=2)
    explicit = parallel_execute(
        ops, inputs, None, _SUM, threaded=True, num_workers=2, accumulate=True
    )
    assert torch.equal(default, explicit)


def test_parallel_execute_empty_raises():
    with pytest.raises(ValueError):
        parallel_execute([], [], None, _SUM, accumulate=True)


# ---------------------------------------------------------------- composites


def _dense_add(*seed_shift):
    return Add(
        *[
            Dense(
                torch.randn(5, 5, generator=torch.Generator().manual_seed(s)).to(
                    COMPLEX64
                ),
                ("M", "N"),
                ("N",),
                ("M",),
            )
            for s in seed_shift
        ]
    )


@pytest.mark.parametrize("threaded", [False, True])
@pytest.mark.parametrize("num_workers", [None, 2])
def test_add_accumulate_matches_plain(threaded, num_workers):
    x = torch.randn(5, generator=torch.Generator().manual_seed(7)).to(COMPLEX64)
    a = _dense_add(1, 2, 3)
    b = _dense_add(1, 2, 3)
    a.threaded, a.num_workers = threaded, num_workers
    b.threaded, b.num_workers, b.accumulate = threaded, num_workers, True
    assert torch.allclose(a(x), b(x), rtol=1e-5, atol=1e-6)
    assert torch.allclose(a.H(x), b.H(x), rtol=1e-5, atol=1e-6)


def test_composites_default_accumulate_false():
    a = _dense_add(1, 2)
    assert a.accumulate is False
    c = Concat(_dense_add(1), _dense_add(2), odim="M")
    assert c.accumulate is False
    s = Stack(_dense_add(1), _dense_add(2), odim_and_idx=("P", 0))
    assert s.accumulate is False


@pytest.mark.parametrize("threaded", [False, True])
@pytest.mark.parametrize("num_workers", [None, 2])
def test_concat_odim_accumulate_is_bitwise(threaded, num_workers):
    """Concat along an output dim reassembles in child order: bitwise invariant."""
    x = torch.randn(5, generator=torch.Generator().manual_seed(8)).to(COMPLEX64)
    y = torch.randn(25, generator=torch.Generator().manual_seed(9)).to(COMPLEX64)
    ops = [
        _dense_add(11),
        _dense_add(12),
        _dense_add(13),
        _dense_add(14),
        _dense_add(15),
    ]
    a = Concat(*ops, odim="M", threaded=threaded, num_workers=num_workers)
    b = Concat(
        *ops, odim="M", threaded=threaded, num_workers=num_workers, accumulate=True
    )
    assert torch.equal(a(x), b(x))
    assert torch.equal(a.H(y), b.H(y))


@pytest.mark.parametrize("threaded", [False, True])
@pytest.mark.parametrize("n", [3, 5])
@pytest.mark.parametrize("num_workers", [None, 2])
def test_stack_odim_forward_accumulate_is_bitwise(n, num_workers, threaded):
    """Output-stacking forward reduce concatenates in child order: bitwise.

    threaded=False must also be covered: the sequential fold must feed
    accumulate_fn post-reduce operands (stack changes the operand signature,
    so raw per-child outputs would merge the wrong axis).
    """
    x = torch.randn(5, generator=torch.Generator().manual_seed(10)).to(COMPLEX64)
    common = dict(threaded=threaded, num_workers=num_workers)
    a = Stack(*[_dense_add(21 + i) for i in range(n)], odim_and_idx=("P", 0), **common)
    b = Stack(
        *[_dense_add(21 + i) for i in range(n)],
        odim_and_idx=("P", 0),
        accumulate=True,
        **common,
    )
    ref = a(x)
    assert ref.shape == (n, 5)
    out = b(x)
    assert out.shape == ref.shape
    assert torch.equal(ref, out)
    # adjoint is a sum reduce: association may differ under chunking -> allclose
    z = torch.randn(n, 5, generator=torch.Generator().manual_seed(11)).to(COMPLEX64)
    assert torch.allclose(a.H(z), b.H(z), rtol=1e-5, atol=1e-6)


# ------------------------------------------------------- create_batched_linop


def _batched_dense(**options):
    weight = torch.randn(10, 3, 7, generator=torch.Generator().manual_seed(12)).to(
        COMPLEX64
    )
    A = Dense(weight, ("B", "M", "N"), ("B", "N"), ("B", "M"))
    x = torch.randn(10, 7, generator=torch.Generator().manual_seed(13)).to(COMPLEX64)
    return create_batched_linop(A, BatchSpec(dict(N=2, M=1)), **options), A, x


def test_create_batched_linop_options_propagate_to_containers():
    B, _, _ = _batched_dense(threaded=True, num_workers=2, accumulate=True)
    containers = [m for m in B.modules() if isinstance(m, (Add, Concat, Stack))]
    assert containers, "expected reassembly containers in the batched tree"
    assert all(m.accumulate is True for m in containers)
    assert all(m.threaded is True for m in containers)
    assert all(m.num_workers == 2 for m in containers)


def test_create_batched_linop_defaults_unchanged():
    B, _, _ = _batched_dense()
    containers = [m for m in B.modules() if isinstance(m, (Add, Concat, Stack))]
    assert all(m.accumulate is False for m in containers)


@pytest.mark.parametrize(
    "options",
    [
        dict(threaded=False, accumulate=True),
        dict(threaded=True, num_workers=2, accumulate=True),
        dict(threaded=True, num_workers=3, accumulate=True),
        dict(threaded=False, num_workers=2, accumulate=True),
    ],
)
def test_create_batched_linop_accumulate_matches_reference(options):
    """Ragged tiles (ceil(7/2)=3 N-tiles with num_workers=2) must stream-accumulate correctly."""
    B_ref, A, x = _batched_dense()
    B_acc, _, _ = _batched_dense(**options)
    y = torch.randn(10, 3, generator=torch.Generator().manual_seed(14)).to(COMPLEX64)
    assert torch.allclose(B_acc(x), B_ref(x), rtol=1e-5, atol=1e-6)
    assert torch.allclose(B_acc.H(y), B_ref.H(y), rtol=1e-5, atol=1e-6)
    assert torch.allclose(B_acc(x), A(x), rtol=1e-5, atol=1e-6)

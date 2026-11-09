# torch.compile Methodology

## Overview

The compiled benchmark suite measures eager vs `torch.compile` behavior for the
same operator registry used by the compile compatibility investigation. It is
separate from the SigPy comparisons and does not change the eager pages.

## Suite levels

- **reduced** — quick compatibility sweep over every operator: forward and
  adjoint, wrapper level (plus inner level for deep-dive operators), default
  compile, CPU and CUDA, 5 warmup / 20 measured iterations. This runs as part of
  `just bench`.
- **full** — complete matrix: all methods, all natural dtypes, wrapper and inner
  levels, `default` and `fullgraph`, 20 warmup / 100 measured iterations, plus
  shape-recompilation, autograd, and pipeline checks. Run it with
  `just bench-compiled`.

Results are written to level-specific files:

- `benchmarks/results/latest/compiled-reduced.json`
- `benchmarks/results/latest/compiled-full.json`

The docs page prefers full-level data when both exist. A compiled-only run
never deletes eager result files.

## Compile units: wrapper vs inner

- **wrapper** — compile the `NamedLinop` call path: `A(x)`, `A.H(y)`, or
  `A.N(x)`. This includes shape checks, named-dimension bookkeeping, logging,
  and device context.
- **inner** — compile the bare static math function: `A.fn`, `A.adj_fn`, or
  `A.normal_fn`. This isolates the numerical kernel from the wrapper.

Functional kernels (`unfold`, `fold`, `grid`, `ungrid`) have no wrapper, so they
are always reported at the inner level.

## Case construction

Each compiled case follows the repository's `linop_input_output` test paradigm:
building a case returns `(linop, input, output)`. The output-shaped tensor is
constructed independently with `torch.randn_like(linop(input))` rather than
being the forward result itself. That keeps adjoint inputs dense and avoids
inheriting forward views.

## Timing

Steady-state per-call medians exclude one-time compilation cost. Inputs are
tiny compared with production workloads, so speedup ratios are directional.

## Graph breaks

Every `default`-config row runs `torch._dynamo.explain` after timing. The result
row records:

- the graph-break count,
- up to three formatted reason/user-stack locations.

`fullgraph` converts graph breaks into loud failures, so its rows record
`graph_breaks = None` rather than a misleading zero. If `explain` itself fails,
the row also records `None` and the monitoring tests fail closed.

Break counts are PyTorch-version dependent and are intended as a monitoring
signal, not a standalone pass/fail criterion.

## Full-suite phase coverage

The full-only phases are deliberately narrower than the sweep:

- CUDA only,
- recompilation is forward-only,
- autograd is deep-dive-only,
- each phase uses the case's first dtype.

Reduced runs do not collect autograd, recompilation, or pipeline results.

## Charts

The generated charts are:

- dtype-labeled compiled speedups on the float-vs-complex page,
- wrapper-vs-inner speedups on the wrapper-vs-inner page,
- pipeline speedups on the chains page.

## Running

```bash
# Reduced compiled suite + report
just bench-compiled-reduced
just bench-report

# Full compiled suite + report
just bench-compiled
just bench-report

# Eager + reduced compiled suite
just bench
just bench-report
```

## Caveats

- One GPU and one PyTorch version per run.
- Dynamic-shape behavior is sampled only by the full suite's recompile phase.
- Complex tensors use complex64; complex128 is not covered.
- The independent-output case construction avoids a Dynamo limitation with
  complex views over real storage; it is a benchmark-harness choice, not a
  change to the functional kernels.

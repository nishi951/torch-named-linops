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

## Class glossary

Every result row gets one effective class. The classes are:

| Class | Meaning |
|---|---|
| `works` | Compiled and ran; output agreed within tolerance; no graph breaks or fallback warnings; speedup at least 0.9x. |
| `works_with_breaks` | Output agreed, but `torch._dynamo.explain` reported one or more graph breaks. |
| `works_with_unknown_breaks` | Output agreed, but graph-break measurement itself failed, so the break count is unknown rather than zero. |
| `compiles_slower` | Output agreed, but the compiled median was below 0.9x eager. |
| `silent_degradation` | Ran and agreed, but a known fallback warning was captured (for example, Inductor falling back for complex operators). This is flagged even when the timing looks normal. |
| `fails_loud` | Raised during eager or compiled execution; the traceback tail is recorded. |
| `fails_silent` | Returned output that disagreed with eager beyond the 1e-3 relative tolerance, or had a shape or non-finite-value mismatch. This is the red-flag class. |
| `n_a` | Not applicable in this environment, for example a CUDA-only path measured on CPU. |
| `blocked_out_of_scope` | Failed because of an excluded dependency path (cupy/SigPy), so it is not counted as a compile verdict. |

Supporting terms:

- **Graph break**: a point where Dynamo cannot continue tracing one graph and splits it into fragments. `fullgraph=True` converts graph breaks into loud failures.
- **Wrapper**: the `NamedLinop` call path, including shape checks and context bookkeeping.
- **Inner**: the bare `fn` / `adj_fn` / `normal_fn` math function.
- **Speedup**: eager median time divided by compiled median time; values below 1.0 mean compiled is slower.

## Caveats

- One GPU and one PyTorch version per run.
- Dynamic-shape behavior is sampled only by the full suite's recompile phase.
- Complex tensors use complex64; complex128 is not covered.
- The independent-output case construction avoids a Dynamo limitation with
  complex views over real storage; it is a benchmark-harness choice, not a
  change to the functional kernels.

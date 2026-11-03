# torch.compile Compatibility Investigation — Findings

Date: 2026-11-03
Branch: `compile-compat-investigation` (off `develop`)
Environment: PyTorch **2.15.0+cu132** (bundled triton 3.9), NVIDIA RTX 3090 (24 GB), CUDA 13.2, Python 3.13.
Machine output: [`report-auto.md`](report-auto.md) (full 726-cell matrix, every row). Raw data: `results/results.json` + `results/results.csv` (every cell carries its effective class and joined recompile count).

---

## Executive summary

We ran 726 compile cells (23 shallow operator variants, 11 deep-dive operator groups including the four triton functional kernels), plus 41 shape-recompile checks, 18 single-op backward checks, and 8 pipeline runs (two chains × four configurations).

**Headline answers**

1. **Can individual operators get faster?** Sometimes. `Dense` (1.30–1.38× median, wrapper-level, default config, CUDA), `Sampling` (1.42×), and simple compositions like `Add` (1.73×) and `Chain` (1.47×) are genuine wins in this harness; `Convolution` is roughly at parity (0.88×). Most small/cheap operators compile *slower* at these shapes (median ratios 0.4–0.9×) because per-call dynamo/compiled-wrapper overhead dominates. On CPU, the triton-free functional kernels are the biggest winners observed anywhere: `unfold` 5.39×, `fold` 5.16×, `ungrid` 2.52×, `grid` 2.34×.
2. **Do longer pipelines compile?** A simple composed pipeline (Dense → Identity, 20-step fixed-point iteration) compiles in default mode and runs **1.36×** faster in the final run, but only by taking **2 graph breaks at the `NamedLinop` wrapper per compiled region**. `fullgraph=True` fails on the wrapper's `logging` call. A realistic MRI-shaped `NUFFT @ Diagonal` pipeline fails to compile in every compiled configuration.
3. **Where does it break?** Two clusters, both expected but now precisely located:
   - **Wrapper-level Python** in `NamedLinop` — `logging` calls inside `SyncContext.__post_init__` (and `_log_transfer`), plus Python shape bookkeeping in `Dense.size` that Dynamo cannot reconstruct. These produce graph breaks in default mode and hard failures under `fullgraph`.
   - **Triton kernels** under Inductor on CUDA — the unfold/fold kernels recompile with a missing `cast` alias (`NameError('cast is not defined')`), and the interp kernels trip `AssertionError: triton_kernel_wrapper_functional was not removed`. On CPU the same functional code compiles cleanly and gets large speedups.

**Silent numeric mismatches: zero.** Every compiled cell that produced output agreed with eager within 1e-3 relative tolerance, with explicit guards for NaN/Inf and shape disagreement (those now classify as `fails_silent`; none occurred in the final run). The failures are loud, with one root-cause-revealing traceback each.

**Silent degradation:** the taxonomy now includes `silent_degradation` and the harness captures Inductor fallback warnings at the compiled call site. **No successful cell exhibited a captured fallback** in this run. Inductor's "does not support code generation for complex operators" warning was seen in overall logs, but per-cell capture did not attribute it to any working cell; treat complex fallback as suspected but unproven (see Root causes).

**Recommended user guidance today:** compile individual `Dense`/`Sampling`/simple-chain operators, or CPU functional kernels; expect little to no gain on small complex operators; avoid `fullgraph=True` until the wrapper logging is guarded; avoid compiling CUDA `NUFFT`/`Interpolate`/unfold/fold paths; for chains, compile the individual linops and leave the loop eager (`compiled-inners` pattern: 1.14× in the final run).

---

## Method

- Harness lives in `dev/compile/compat-investigation/` and is re-runnable phase by phase: `uv run python dev/compile/compat-investigation/harness.py --phase {smoke,deep,recompile,autograd,pipeline,all}`.
- Each cell: eager reference + median-of-100 timing (20 warmup; 10 iterations when one call > 0.5 s), then `torch._dynamo.reset()`, then `torch.compile` with default or `fullgraph=True`, then numerics vs eager (`max_rel_diff` gate 1e-3, NaN/Inf and shape disagreement always classify as disagreement), graph-break count/locations via `torch._dynamo.explain`, fallback-warning capture, and median compiled timing.
- Matrix axes: every registry operator × fwd/adj/normal × CUDA/CPU × float32 (complex64 additionally for designated complex-native groups) × wrapper/inner compile unit × eager/default/fullgraph.
- Deep-dive set: NUFFT, FastSamplingNUFFT, SamplingNUFFT, ToeplitzPSF, Interpolate, Convolution, FFT, and the functional `unfold`/`fold`/`grid`/`ungrid` kernels.
- `BreakpointLinop` was excluded deliberately (it invokes `pdb`; an automated run would block). It is the only operator not measured.

### Harness corrections made during the run (for trust)

Several measurement bugs were found and fixed with regression tests, because each changed conclusions:

1. Dynamo "not supported" fullgraph errors were first misclassified as n/a; the n/a rule now only covers CPU-with-CUDA/Triton unavailability. (Without the fix, 15 real failures looked like skips.)
2. Recompile counts initially accumulated across cases; the metric now counts only graphs built after the first call, handles graph-breaking functions, and is pinned by same-shape/two-shape/graph-breaking tests.
3. Functional cases were initially never executed (wrong builder for level `inner`), which made them look like eager crashes; fixed and re-run.
4. The numeric comparator originally failed open on NaN and shape changes (`max()` with NaN, broadcasting); it now treats shape disagreement and non-finite values as disagreement. Regression tests cover NaN, shape-change, and incompatible-shape inputs.
5. Effective taxonomy (`works_with_breaks`, `compiles_slower`, `silent_degradation`, plus failure classes) is now computed and serialized per cell; recompile counts are joined onto matching cells in `results.json`.

---

## Results at a glance

Effective classes (serialized per cell, 726 cells):

| class | cells |
|---|---|
| works | 398 |
| works with breaks | 29 |
| compiles, slower (<0.9×) | 196 |
| fails loudly | 103 |
| fails silently | **0** |
| silent degradation | 0 |
| n/a | 0 |

Failures by device/config: CUDA 77 / CPU 26; by mode `fullgraph` 69, `default` 34, `eager` **0**. Raw `cell_class` values are `works` 623 / `fails_loud` 103 (the difference from effective classes is the taxonomy overlay).

### What compiles cleanly at the wrapper level (CUDA, default)

Among tested methods: Convolution, Dense, DenseSimple, Diagonal, FFT, Identity, Zero, ShapeSpec, Pad, PadDim, Rearrange, Scalar, SimpleLinop, Sleep, ToDevice, ToeplitzPSF, plus clean rows of Sampling/Repeat/Truncate/SumReduce (some methods of those ops take a break or fall to a different path).

### Operators that fail under `torch.compile` (CUDA, default)

| operator | error signature | mechanism |
|---|---|---|
| `ArrayToBlocks`, `BlocksToArray` | `NameError('cast is not defined')` in generated triton kernel | unfold/fold kernels reference module-level alias `from .casting import scalar_cast as cast`, absent when Inductor recompiles the kernel source |
| `functional.unfold`, `.fold` | same `cast` NameError | same triton integration issue |
| `functional.grid`, `.ungrid` | `triton_kernel_wrapper_functional` not removed | interp kernel wrapper survives functionalization |
| `Interpolate`, `NUFFT` | `AssertionError: triton_kernel_wrapper_functional was not removed` (Inductor) | same interp-kernel wrapper issue |
| `Concat` | `NotImplementedError: currently can't reconstruct arbitrary frozen dataclass instances` | `Dense.size()` does `self.weightshape.index(dim)` inside `check_input_shape`; Dynamo cannot reconstruct `NamedDimension` frozen dataclasses while tracing |
| `Stack` | same dataclass error on fwd/adj; `normal` compiles | same mechanism, method-dependent path |

On CPU, `Interpolate`, `NUFFT`, and all four functional kernels compile fine (2.3–5.4× speedups); the remaining CPU failures are the same `Concat`/`Stack` dataclass issue plus `ArrayToBlocks`/`BlocksToArray` adjoints (`RuntimeError: Tensor must have a last dimension of size 2` from `view_as_complex` on fake tensors). This is strong evidence that the CUDA failures are triton-integration-specific, not logic bugs in the operators.

### `fullgraph=True`

69 failures. On CUDA, every operator that observes a CUDA stream path fails at `SyncContext.__post_init__`'s `logger.debug` (`logging.Logger method not supported for non-export cases`) — Add, Chain, Repeat, Sampling, SamplingNUFFT, FastSamplingNUFFT, SumReduce, Truncate, plus the triton/dataclass failures above. `Sleep` additionally fails on its `torch.cuda.stream(...)` context manager. Fullgraph is effectively unusable through the wrapper today, and the single blocking call is one `logger.debug`. FFT, ToeplitzPSF, Convolution, Dense and the simple ops do pass fullgraph.

### Speedups (wrapper, default, CUDA; median per operator)

| operator | median speedup |
|---|---|
| Add | 1.73× |
| Chain | 1.47× |
| Sampling | 1.42× |
| DenseSimple | 1.38× |
| Dense | 1.30× |
| Pad | 1.02× |
| SumReduce | 0.95× |
| ToDevice | 0.93× |
| Truncate | 0.91× |
| Repeat | 0.90× |
| Convolution | 0.88× |
| SimpleLinop | 0.87× |
| Sleep | 0.79× |
| PadDim | 0.76× |
| ShapeSpec | 0.73× |
| Diagonal | 0.73× |
| Identity | 0.73× |
| Scalar | 0.72× |
| Rearrange | 0.71× |
| Zero | 0.71× |
| ToeplitzPSF | 0.69× |
| FastSamplingNUFFT | 0.68× |
| Stack | 0.67× |
| SamplingNUFFT | 0.64× |
| FFT | 0.43× |

(Small test shapes flatter large operators and penalize tiny ones; these ratios are indicative, not a benchmark claim. `Sleep`'s ratio is not meaningful — eager sleeps; ignore it.)

Notable inner-level win: `fast_sampling_nufft` forward 1.7×; functional kernels on CPU as above.

---

## Recompilation

41 checks, 21 with a batch-shaped family that eager accepts:

- 12 recompiled **exactly once** after a 2× batch change; 2 recompiled 0 times (`Identity`, `ShapeSpec`); 7 never reached measurement because the first compiled call already failed.
- Metric: graphs built for calls after the first (a graph-breaking first call can build multiple fragments without that being a recompile). Same-shape, two-shape, and graph-breaking same-shape cases are pinned by tests.
- 96 matrix cells now carry joined recompile counts (default-config rows matched by op+level); 84 are non-zero.
- `torch._dynamo.utils.counters["stats"]["unique_graphs"]` is the only useful counter on this torch build; the plan's `recompilation_reasons`/`graph_breaks` keys do not exist.

Users looping over varying batch sizes should call `torch._dynamo.mark_dynamic` on the batch dim (or accept one recompile) once the triton blockers are fixed.

---

## Autograd

18 single-op backward checks (deep-dive ops, CUDA):

- **10 work** with matching gradients: FFT, Convolution, FastSamplingNUFFT, SamplingNUFFT, ToeplitzPSF (wrapper and inner).
- **4 fail only in compiled backward**: Interpolate, NUFFT (wrapper and inner) — same triton/Inductor path as forward.
- **4 are not differentiable even eagerly**: `functional.unfold`, `.fold`, `.grid`, `.ungrid` (`element 0 of tensors does not require grad and does not have a grad_fn`). Eager-phase failure; not a compile regression, but it blocks using these kernels in training loops at all.

No gradient mismatches were observed: where backward worked, compiled and eager grads agreed.

---

## Pipeline

Two chains, 20-step fixed-point iteration `x ← x − α·Aᴴ(Ax − b)`:

| pipeline | config | result | speedup | breaks |
|---|---|---|---|---|
| simple (Dense→Identity) | eager | works | — | — |
| simple | default | works | 1.36× | 2 (wrapper logging) |
| simple | fullgraph | **fails** at `SyncContext` logging | — | — |
| simple | compiled-inners (eager loop, compiled operators) | works | 1.14× | — |
| nufft (NUFFT→Diagonal) | eager | works | — | — |
| nufft | default | **fails** (`triton_kernel_wrapper_functional`) | — | — |
| nufft | fullgraph | **fails** (triton) | — | — |
| nufft | compiled-inners | **fails** (compiling the NUFFT operator itself) | — | — |

The wrapper graph breaks are the exact `logging` calls in `namedlinop.py` (`SyncContext.__post_init__`, lines 711/719; `_log_transfer` line 29-31). The compiled-inners pattern is the only currently reliable way to get compiled work into an eager pipeline, and only for operators that compile individually.

---

## Implicated root causes

1. **Wrapper Python executes during tracing.** `NamedLinop.forward` runs `check_input_shape` (Python loops over named dims, `Dense.size()` list indexing over frozen dataclasses), `SyncContext` event bookkeeping, and CUDA logging before/around the math. Any pipelined operator inherits these graph breaks; `fullgraph` converts them to errors.
2. **Triton kernels are not Inductor-transparent.** Module-level aliases (`cast`) vanish from Inductor's recompiled kernel source; the interp kernels' wrapper is not functionalization-friendly. CUDA-only: the torch fallback path compiles well (and fast) on CPU.
3. **Complex tensors.** Inductor warns that it does not generate code for complex operators, and fake-tensor tracing of `view_as_complex` fails in the fold/adjoint path. Most complex paths either fall back (parity) or fail; measured FFT is 0.43×. Per-cell warning capture did not attribute an explicit fallback event to any working cell, so this remains suspected rather than classified as silent degradation.
4. **Per-call overhead dominates small operators.** The compiled wrapper's dispatch/guard cost exceeds kernel savings below roughly Dense-sized work; gains show up on matmul-heavy or CPU-fused work.
5. **Differentiability gaps.** The functional kernels are not autograd-capable eagerly; NUFFT/Interpolate backward additionally breaks under compile.

---

## Cheap fixes (drafted as issues)

Issue drafts live in `issues/`. Ranked by leverage per unit of work:

1. **`issues/issue-1-logging-graph-breaks.md`** — Guard/remove the `logger.debug` calls in `SyncContext.__post_init__` and `_log_transfer` when tracing (`torch.compiler.is_compiling()` early-out or module-level `logger.isEnabledFor` check executed once). Removes 29 observed break-cells, unblocks `fullgraph` for ~8 operators, and removes the 2 pipeline breaks. Highest leverage; small change.
2. **`issues/issue-2-triton-cast-alias.md`** — Stop referencing the module-level `cast` alias from triton kernel bodies in `functional/_unfold/`; use `tl.cast`/direct bitcast or pass the function via closure/global explicitly. Unblocks CUDA unfold/fold/ArrayToBlocks/BlocksToArray (grid/ungrid additionally need issue 3).
3. **`issues/issue-3-triton-wrapper-functionalization.md`** — Investigate the `triton_kernel_wrapper_functional was not removed` failure for grid/ungrid; likely needs `torch.library.custom_op` with fake/meta implementations or an alternate kernel-launch pattern. Unblocks Interpolate/NUFFT compiled forward and backward (the largest single capability gap).
4. **`issues/issue-4-dense-size-frozen-dataclass.md`** — Replace `self.weightshape.index(dim)`-style shape lookups in `Dense.size()` with a precomputed name→size mapping (plain ints/strings), avoiding frozen `NamedDimension` reconstruction during tracing. Unblocks Concat/Stack and any composition that calls `check_input_shape` mid-graph.
5. **`issues/issue-5-functional-kernel-autograd.md`** — Decide and implement autograd support for `functional/_unfold` and `_interp` kernels (custom `torch.autograd.Function` or documented non-differentiability). Required before these kernels can appear inside training loops at all.

Not cheap / not recommended now: making `NamedLinop` itself a fully traceable container (the wrapper-vs-inner split suggests the ergonomic path is compiling `fn`/`adj_fn` after the above fixes, not re-architecting `forward`).

---

## Limits of this investigation

- One GPU, one PyTorch version (2.15.0+cu132), one triton version (3.9). Dynamic-shape behavior was sampled only via 2× batch changes.
- Small operator shapes keep runs fast and deterministic; speedup ratios are shape-dependent and should not be extrapolated to production sizes or other backends.
- Pipeline uses synthetic chains shaped after `tutorials/mri_reconstruction.py`, not the tutorial end-to-end (which includes plotting/marimo code).
- Timing excludes one-time compilation cost by design (warmup + median), i.e. these are steady-state per-call ratios. Pipeline timing uses 1 warmup + 3 repeats (matrix cells use the pinned 20/100 protocol).
- Matrix dtypes: float32 for all non-complex-native operators; complex64 for the designated complex-native groups (FFT, NUFFT family, Toeplitz, Interp, ArrayToBlocks/BlocksToArray) and additionally for Convolution.

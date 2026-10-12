# How Named Dimensions Propagate

Every `NamedLinop` carries named dimensions (`ishape`, `oshape`, plus auxiliary shapes like `weightshape` or `batch_shape`). Those names are not decorations: they are what composition (`@`), adjoints/normal operators (`.H`/`.N`), splitting, and shape-inference all operate on. This page explains the rules they follow.

For the data structures themselves (`NamedDimension`, `NamedShape`, `NamedDimCollection`), see [Named Abstractions](named_abstractions.md). For the config flag that controls automatic inference in containers, see [Configuration Options](config.md).

## The mental model

**Dimensions are identified by their name.** Equality and hashing are purely string-based:

```python
from torchlinops import Dim
from torchlinops.nameddim import NamedDimension as ND

nx = ND("Nx")  # name "Nx", index 0
nx1 = ND("Nx", 1)  # name "Nx1" — distinguished variant of "Nx"
print(nx == "Nx", nx1 == "Nx1", nx != nx1)
```

```
True True True
```

Strings like `"Nx"` and `"Nx1"` are converted to `NamedDimension`s automatically wherever shapes are accepted (`ND.infer`), so `Dim("MN")`, `("M", "N")`, and interleaved strings all work.

**Each linop owns a shared pool of dimensions.** A linop's `NamedShape` registers every shape it knows about — `ishape`, `oshape`, and auxiliary shapes (`weightshape`, `batch_shape`, `input_shape`, `axes_lengths`, ...) — against a *single* flat list of dimension objects. A dim that appears in two registered shapes is one shared entry, so an update through one shape immediately appears in the other:

```python
import torch
from torchlinops import Dense, Dim

w = torch.randn(3, 7)
A = Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
print(A.shape, "| weightshape:", A._shape.weightshape)

A.oshape = ("Q",)  # rename M -> Q via the oshape ...
print(A.shape, "| weightshape:", A._shape.weightshape)  # ... renamed in weightshape too
```

```
(N,) -> (M,) | weightshape: (M, N)
(N,) -> (Q,) | weightshape: (Q, N)
```

This is by design: auxiliary metadata should never be left pointing at a dim the main shapes no longer mention.

**Pools are per-linop.** Renames never leak between different linops; when two linops are composed, their dimensions are *matched* by name, not linked.

## Special dimensions

Two wildcard names have matching semantics instead of concrete meanings:

| Token | Matches | Typical origin |
|---|---|---|
| `...` | zero or more dimensions | default batch shapes (`FFT`, `Sampling`, ...), empty shapes (`Identity`, `Scalar`) |
| `()` | exactly one dimension of any name | `Dense`'s inferred shapes |

Numbered anonymous dims `(1)`, `(2)`, ... are also positional wildcards — distinct from each other, matchable to any single dim. They are what auto-inference produces.

## What happens at construction

Constructors fill in defaults when you don't specify shapes:

- **Diagonal shortcut**: passing only an `ishape` (or a single "io" shape) makes `oshape = ishape` — `Diagonal(d, ioshape=Dim("N"))` is `(N,) -> (N,)`.
- **Empty shapes**: `Identity()` and `Scalar(c)` have no dimensions: `(...,)` on both sides.
- **Batch dims default to `...`**: `FFT(ndim=1)` is `FFT((..., Nx) -> (..., Kx))`.
- **Convention-specific defaults**: `FFT` names its grids `Nx`/`Ny`/`Nz` and `Kx`/`Ky`/`Kz`; omitting `grid_shapes` uses this convention.
- **Dense auto-inference**: `Dense(w)` names every weight axis with an anonymous dim: the last weight axis is the input, the second-to-last is the output, and the rest are batch dims shared between input and output:

```python
import torch
from torchlinops import Dense

w = torch.randn(3, 7)
A = Dense(w)  # (M=3 out, N=7 in)
print(A)

W = torch.randn(2, 5, 3)  # (batch, out, in)
B = Dense(W)  # batched matmul
print(B)
```

```
Dense(((1),) -> ((),))
Dense(((), (2)) -> ((), (1)))
```

- **Operators that "grow" a dimension** name the new (output) dim as the incremented input dim: `Pad` maps `(..., Nx) -> (..., Nx1)`, `Convolution(k)` maps `(..., Nx, Ny) -> (..., Nx1, Ny1)`. Incrementing (`Nx` → `Nx1` → `Nx2`) skips names already taken in the reference shapes (`next_unused`).

## Rules at a glance

| Trigger | Effect on dimensions | Why |
|---|---|---|
| Constructing with partial shapes | Defaults fill in (diagonal shortcut, `...` batches, empty shapes) | Keep constructors lightweight |
| `Dense(w)` without shapes | Anonymous `()` dims; last weight axis = input, second-to-last = output, rest = batch | Inferring the only sensible contract a matrix has |
| Setting `linop.ishape`/`oshape` | Length-checked; equal-name dims rename across **all** of that linop's shapes; wildcard assignments replace only the assigned shape; `...` splices matched dims in | One shared source of truth per linop |
| Assigning a wildcard to a shape | Hides the concrete name in *that shape only*, not in auxiliary shapes | The wildcard is an intentional "no name here" |
| `Chain(...)` | Derived shape is *dynamic*: first linop's `ishape`, last linop's `oshape`; setting `chain.ishape` renames inside the first linop; with `config.shape_inference`, each linop's `ishape` is resolved forward from the previous `oshape` | Composition defines the outer contract |
| `Add(...)` | Operands must agree; with `shape_inference`, a consensus shape wins (concrete beats wildcards, higher index beats lower, e.g. `N1` over `N`) and operands are renamed onto it | Sums require identical domains/codomains |
| `A.H` | `ishape` and `oshape` swap; auxiliary shapes carry over | Definition of the adjoint |
| `A.N` (default) | "Diagonal" dims (in both `ishape` and `oshape`) keep their name; others are regenerated with `next_unused`; then the default normal **increments** `oshape` again (`Nx` → `Nx1`) | $A^H A$ maps the domain to a *fresh copy* of itself: pre-image and post-image dims must not collide |
| `A.N` with operator-specific overrides | Many operators replace the default (see below) | Mathematical structure allows better versions |
| Splitting / device transfer | Names pass through untouched; only slice positions and data move | Tiling must not change the operator's meaning |
| Functional layer (`torchlinops.functional`) | No names at all — plain tensors | Names live in the operator layer |

## Setting shapes on an existing linop

Assigning to `ishape`/`oshape` (or any auxiliary shape) on a linop goes through the same machinery:

1. **Length compatibility** is checked first (modulo `...`); incompatible updates raise `ValueError`.
2. **Concrete → concrete**: a rename, propagated to every shape of *that linop* that mentions the dim (see the shared-pool example above). Within one assignment, a dim may not be renamed two different ways.
3. **Concrete → wildcard**: assigning a wildcard hides the name only in the assigned shape:

```python
A2 = Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
A2.oshape = ("...",)
print(A2.shape, "| weightshape:", A2._shape.weightshape)
```

```
(N,) -> ('...',) | weightshape: (M, N)
```

4. **`...` splicing**: assigning concrete dims where the shape has `...` fills them in place.
5. Omitted dims must stay consistent: every concrete dim in the old shape must receive exactly one new value, or the update fails.

## Containers

### `Chain`

A chain's shape is **dynamic**: `chain.ishape` *is* `chain.linops[0].ishape` and `chain.oshape` *is* `chain.linops[-1].oshape`, rebuilt on every access. Consequently, setting `chain.ishape = ...` renames dimensions inside the first linop (and chain printing always reflects the constituents):

```python
from torchlinops import FFT, Dim

F = FFT(ndim=1, grid_shapes=(Dim("M"), Dim("K")), batch_shape=Dim())
C = F @ Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
C.ishape = ("Q",)
print(C.shape)
print(C.linops[0])
```

```
(Q,) -> (K,)
Dense((Q,) -> (M,))
```

At construction a chain validates every interface (`output of i == input of i+1`) and raises `ValueError` naming the mismatch otherwise. With `config.shape_inference=True`, wildcards in each subsequent linop's `ishape` are resolved from the previous linop's `oshape` — see [Configuration Options](config.md) for worked examples. Resolution is forward-only and only abstracts *unspecified* dims: concrete dims already present are left alone (they are the ones that define matching).

### `Add`

All operands of an `Add` must have equal `ishape`/`oshape` (checked at construction). With `shape_inference`, a consensus is computed with `max_shape` — concrete dims beat wildcards; among same-named dims the higher index wins (`N1` over `N`) — and every operand is renamed onto it. In normals, each diagonal/cross term is additionally renamed onto the canonical `A^H A` shape via the same mechanism.

Quirk to be aware of: the normal of an `Add` shares its shape object with its first constituent term, so renames applied through that wrapper also rename the first term.

### Renaming operators

`ShapeSpec` is an `Identity`-like operator whose only job is to bridge differently-named shapes, e.g. `(N,) -> (N1,)`. Insert it into a chain when two operators do the same thing but speak different names.

## Adjoint and normal

### Adjoint (`.H`)

The adjoint swaps `ishape` and `oshape` of a fresh copy; auxiliary shapes carry over as-is. Because `.H` builds on copies, its renames never leak back into the original.

### Normal (`.N`) — the increment rule

The normal operator $A^H A$ maps the domain back to itself — but the library keeps the "outer" copy of the domain dims under distinct names so nothing inside the folded chain can be confused between pre-image and post-image. The default normal machinery does this in two steps:

1. `NamedShape.N`: dims appearing in both `ishape` and `oshape` (genuinely diagonal) keep their name; every other `oshape` dim is regenerated with `next_unused`.
2. The default `NamedLinop.normal` then increments the new `oshape` against the final `ishape`, so even a "diagonal" default becomes `Nx → Nx1`:

```python
A = Dense(w, Dim("MN"), ishape=Dim("N"), oshape=Dim("M"))
print(A.shape)
print(A.N.shape)
```

```
(N,) -> (M,)
(N,) -> (N1,)
```

Anonymous dims follow the same rule: `Dense(w_tall).N` prints as `Dense(((1),) -> ((2),))`.

### Operator-specific normal overrides

Many operators replace the default with something mathematically better:

| Operator | Normal behavior |
|---|---|
| `Diagonal` | $D^H D = \|w\|^2 \odot I$: keeps names identical, `Diagonal((N,) -> (N,))`. |
| `FFT`, `Rearrange`, `Identity` | Return `Identity` (unitary / reindexing) or `inner` directly. |
| `ToDevice` | Normal becomes `Identity` with default (unnamed) shapes — names are dropped. |
| `Dense` | Rebuilds the weight of $W^H W$ by "surgery" on `weightshape`: a dim appearing in `ishape` and `weightshape` but not `oshape` is incremented; a dim appearing in all three survives as-is; the result of the example above is `weightshape: (M, N)` → `(N1, N)`. |
| `SumReduce` | Dims summed over get renamed (`N` → `N1`) in the normal's oshape; `axes_lengths` metadata follows automatically through the shared pool. |
| `Convolution` | Increments relative to the input grid: `(..., Nx) -> (..., Nx1)`. |
| `Pad` / `Truncate` | Sandwich shapes are spliced surgically — everything untruncated comes from the inner operator, only the padded/truncated dim carries its own name. |
| `Stack` | Introduces the new stacking dim, which must not already exist in that shape. |
| `Concat` | Interfaced dims are standardized with the same `max_shape` consensus as `Add`. |

If you ever see an unexpected `Nx1` in a print, it is almost always this increment rule in one of the above places.

## What never renames dims

- **Splitting** (`split_linop`, `create_batched_linop`): tiles slice along existing names and keep them (`Dense((B, N) -> (B, M))` per tile).
- **Device transfer** (`linop.to(...)`, `ToDevice` wrapping): moves tensors and operators, not names.
- **The functional layer** (`torchlinops.functional.*`): works on plain tensors with no named-dim semantics.

## Gotchas worth knowing

- A trailing `...` can match **zero** dimensions: `isequal(("...", "A"), ("A",))` holds.
- The default `.N` increments names even when the operator feels diagonal (`Dense` above shows `(N,) -> (N1,)`); only operators with explicit normal overrides keep names symmetric (`Diagonal`, `FFT`, ...).
- Comparing shapes to raw strings is safe per-dim (`dim == "Nx"`), but a tuple shape `("...",)` does not equal the bare string `"..."` — singleton-registered shapes do.
- `.H`/`.N` are cached; renamed dims don't invalidate the cache — `linop.reset_adjoint_and_normal()` after renames, or disable `cache_adjoint_normal` (see [Configuration Options](config.md)).
- Setting one shape can rename another (shared pool) — e.g. after `A.oshape = ("Q",)` above, the weight matrix is now "the `(Q, N)` matrix", not the `(M, N)` one.

import marimo

__generated_with = "0.1.67"
app = marimo.App()


@app.cell
def __():
    from copy import copy, deepcopy
    from inspect import signature

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn as nn
    import sigpy as sp
    import sigpy.mri as mri
    from torchkbnufft import KbNufft, KbNufftAdjoint
    from einops import rearrange
    return (
        KbNufft,
        KbNufftAdjoint,
        copy,
        deepcopy,
        mo,
        mri,
        nn,
        np,
        rearrange,
        signature,
        sp,
        torch,
    )


@app.cell
def __(
    KbNufft,
    KbNufftAdjoint,
    copy,
    logger,
    nn,
    rearrange,
    signature,
    torch,
):
    class NamedLinop(nn.Module):
        def __init__(self, ishape, oshape):
            """ishape and oshape are symbolic, not numeric
            They also change if the adjoint is taken (!)
            """
            super().__init__()
            self.ishape = ishape
            self.oshape = oshape

            self._adj = None
            self._normal = None

            self._suffix = ''

        # Change the call to self.fn according to the data
        def forward(self, x: torch.Tensor):
            return self.fn(x)

        # Probably don't override these
        @property
        def H(self):
            """Adjoint operator"""
            if self._adj is None:
                _adj = copy(self)
                # Swap functions
                _adj.fn, _adj.adj_fn = self.adj_fn, self.fn
                # Swap shapes
                _adj.ishape, _adj.oshape  = self.oshape, self.ishape
                _adj._suffix += '.H'
                self._adj = _adj
            return self._adj

        @property
        def N(self):
            """Normal operator (is this really necessary?)"""
            if self._normal is None:
            #     _normal = copy(self)
            #     _normal._suffix += '.N'
            #     self.normal = _normal
            # return self._normal
                _normal = copy(self)
                _normal.fn = self.normal_fn
                _normal.adj_fn = self.normal_fn
                def new_normal(x, *args, **kwargs):
                    x = self.normal_fn(x, *args, **kwargs)
                    return self.normal_fn(x, *args, **kwargs)
                _normal.normal_fn = new_normal
                _normal.ishape = self.ishape
                _normal.oshape, _normal.ishape = self.ishape, self.ishape
                _normal._suffix += '.N'
                self._normal = _normal
            return self._normal

        # Override these
        def fn(self, x: torch.Tensor, /, **kwargs):
            """Placeholder for functional forward operator.
            Non-input arguments should be keyword-only
            self can still be used - kwargs should contain elements
            that may change frequently (e.g. trajectories) and can
            ignore hyperparameters (e.g. normalization modes)
            """
            return x

        def adj_fn(self, x: torch.Tensor, /, **kwargs):
            """Placeholder for functional adjoint operator.
            Non-input arguments should be keyword-only"""
            return x

        def normal_fn(self, x: torch.Tensor, /, **kwargs):
            """Placeholder for efficient functional normal operator"""
            return self.adj_fn(self.fn(x, **kwargs), **kwargs)

        def split(self, ibatch, obatch):
            """Transform internal data so that `forward`
            performs a split version of the linop
            ibatch: tuple of slices of same length as ishape 
            obatch: tuple of slices of same length as oshape
            """
            raise NotImplementedError(f'Linop {self.__class__.__name__} cannot be split.')

        def _flatten(self):
            """Get a flattened list of constituent linops for composition"""
            return [self]

        def _compose(self, inner):
            """Do self AFTER inner"""
            before = inner._flatten()
            after = self._flatten()
            return Chain(*(after + before))

        def __matmul__(self, other):
            return self._compose(other)

        def __rmatmul__(self, other):
            return other._compose(self)

        def __repr__(self):
            """Helps prevent recursion error caused by .H and .N"""
            return f'{self.__class__.__name__ + self._suffix}({self.ishape} -> {self.oshape})'


    class Chain(NamedLinop):
        def __init__(self, *linops):
            super().__init__(linops[-1].ishape, linops[0].oshape)
            self.linops = list(linops)
            self.signatures = [signature(linop.fn) for linop in self.linops]
            self._check_signatures()

        def _check_signatures(self):
            seen = set()
            for sig in self.signatures:
                for param in sig.parameters.values():
                    if param.name in seen:
                        logger.debug(
                            f'{param.name} appears more than once in linop chain.'
                        )

        def _parse_kwargs(self, kwargs):
            all_linop_kwargs = []
            for sig in self.signatures:
                linop_kwargs = {}
                for param in sig.parameters.values():
                    if param.name in kwargs:
                        linop_kwargs[param.name] = kwargs[param.name]
                all_linop_kwargs.append(linop_kwargs)
            return all_linop_kwargs

        def fn(self, x: torch.Tensor, /, **kwargs):
            all_linop_kwargs = self._parse_kwargs(kwargs)
            for linop, kw in zip(reversed(self.linops),
                                 reversed(all_linop_kwargs)):
                x = linop(x, **kw)
            return x

        def adj_fn(self, x: torch.Tensor, /, **kwargs):
            all_linop_kwargs = self._parse_kwargs(kwargs)
            for linop, kw in zip(self.linops, all_linop_kwargs):
                x = linop.adj_fn(x, **kw)
            return x

        def normal_fn(self, x: torch.Tensor, /, **kwargs):
            return self.adj_fn(self.fn(x, **kwargs))

        def _flatten(self):
            return self.linops

        def __repr__(self):
            return f'{self.__class__.__name__}({",".join(repr(linop) for linop in self.linops)})'


    class Broadcast(NamedLinop):
        """Return a rearrange matching batched ishape to oshape.
        Basically broadcast to each other
        """
        def __init__(self, ishape, oshape):
            super().__init__(ishape, oshape)
            self.ishape_str = ' '.join(ishape)
            self.oshape_str = ' '.join(oshape)

        def forward(self, x):
            return self.fn(x, self.ishape_str, self.oshape_str)

        @classmethod
        def fn(cls, x, ishape_str, oshape_str):
            return rearrange(x, f'... {ishape_str} -> ... {oshape_str}')

        @classmethod
        def adj_fn(cls, x: torch.Tensor, ishape_str, oshape_str):
            return rearrange(x, f'... {oshape_str} -> ... {ishape_str}')

        def split(self, ibatch, obatch):
            return self # Literally don't change anything

    def get2dor3d(im_size, kspace=False):
        if len(im_size) == 2:
            im_dim = ('kx', 'ky') if kspace else ('x', 'y')
        elif len(im_size) == 3:
            im_dim = ('kx', 'ky', 'kz') if kspace else ('x', 'y')
        else:
            raise ValueError(f'Image size {im_size} - should have length 2 or 3')
        return im_dim

    class NUFFT(NamedLinop):
        def __init__(
            self,
            trj,
            im_size,
            trj_batch_shape,
            norm='ortho',
            kbnufft_kwargs=None,
            # Add more stuff for e.g. grog
            grog_normal=True,
            grog_config=None,
        ):
            """
            trj: (... d k) in -pi to pi (tkbn-style)
            """
            ishape = get2dor3d(im_size)
            oshape = trj_batch_shape + ('D', 'K')
            super().__init__(ishape, oshape)
            self.trj = trj
            self.im_size = im_size

            # KbNufft-specific
            self.norm = norm
            kbnufft_kwargs = kbnufft_kwargs if kbnufft_kwargs is not None else {}
            self.nufft = KbNufft(im_size, **kbnufft_kwargs)
            self.nufft_adj = KbNufftAdjoint(im_size, **kbnufft_kwargs)

        def forward(self, x: torch.Tensor):
            return self.fn(x, self.trj, self.im_size)

        def fn(self, x, /, trj, im_size):
            y = self.nufft(x, trj, norm=self.norm)
            return y

        def adj_fn(self, x, /, trj):
            y = self.nufft_adj(x, trj, norm=self.norm)
            return y

        def normal_fn(self, x, /, trj):
            ...

        def __repr__(self):
            return f'{self.__class__.__name__}({self.ishape} -> {self.oshape})'

    class SENSE(NamedLinop):
        def __init__(self, mps):
            im_size = mps.shape[1:]
            im_shape = get2dor3d(im_size, kspace=False)
            super().__init__(im_shape, ('C', *im_shape))
            self.mps = mps

        def forward(self, x):
            return self.fn(x, self.mps)

        def fn(self, x, /, mps):
            return x * mps

        def adj_fn(self, x, /, mps):
            return torch.sum(x * torch.conj(mps), dim=0)

        def split(self, ibatch, obatch):
            """Split over coil dim only"""
            for islc, oslc in zip(ibatch, obatch[1:]):
                if islc != oslc:
                    raise IndexError(f'SENSE currently only supports matched image input/output slicing.')

            return self.__class__(self.mps[obatch])


    class Repeat(NamedLinop):
        """Unsqueezes and expands a tensor along dim
        """
        def __init__(self, n_repeats, dim):
            super().__init__(tuple(), tuple())
            self.n_repeats = n_repeats
            self.dim = dim

        def forward(self, x):
            return self.fn(x)

        def fn(self, x, /):
            x = x.unsqueeze(self.dim)
            # print(x)
            return torch.repeat_interleave(x, self.n_repeats, dim=self.dim)

        def adj_fn(self, x, /):
            return torch.sum(dim=self.dim, keepdim=False)


    class Diagonal(NamedLinop):
        def __init__(self, weight, ishape, oshape):
            super().__init__(ishape, oshape)
            self.weight = weight

        def forward(self, x):
            return x * self.weight
        
        def fn(self, x, /, weight):
            return x * weight

        def adj_fn(self, x, /, weight):
            return x * torch.conj(weight)

        def normal_fn(self, x, /, weight):
            return x * torch.abs(weight) ** 2

    class Dense(NamedLinop):
        ...


    class Batch(NamedLinop):
        def __init__(self, linop, **batch_sizes):
            self.linop = linop
            self.batch_sizes = batch_sizes
            self._bat

        def forward(self, x: torch.Tensor):

            ...

    return (
        Batch,
        Broadcast,
        Chain,
        Dense,
        Diagonal,
        NUFFT,
        NamedLinop,
        Repeat,
        SENSE,
        get2dor3d,
    )


@app.cell
def __(Diagonal):
    d = Diagonal(1, 'a', 'b')
    print(d.H)
    print(d.N)

    return d,


@app.cell
def __(NUFFT, Repeat, SENSE, mri, np, rearrange, torch):
    # B, Nx, Ny, T, K, D, C all defined
    B = 5
    Nx = 64
    Ny = 64
    C = 12
    T = 10
    K = 100
    D = 2
    num_interleaves = 16
    trj = mri.spiral(
        fov=1,
        N=Nx,
        f_sampling=0.2,
        R=1,
        ninterleaves=num_interleaves,
        alpha=1.5,
        gm=40e-3,
        sm=100,
    )
    trj = rearrange(trj, '(r k) d -> r k d', r=num_interleaves)
    # print(trj.shape)

    x = torch.randn((Nx, Ny), dtype=torch.complex64)
    x_dims = ('B', 'Nx', 'Ny')
    # Convert sigpy trj to tkbn trj
    trj = torch.from_numpy(trj)
    trj = rearrange(trj, '... k d -> ... d k')
    trj = trj * 2 * np.pi

    mps = torch.randn((C, Nx, Ny), dtype=torch.complex64)
    F = NUFFT(trj, im_size=(Nx, Ny), trj_batch_shape=('R',))
    S = SENSE(mps)
    R = Repeat(n_repeats=num_interleaves, dim=0)
    # BC = Broadcast(('B', 'Nx', 'Ny'), ('B', '1', 'Nx', 'Ny'))
    A = F @ R @ S
    # print(A.batchable_dims())
    # TODO
    # A = Batch(A, B=3, C=1, T=2)

    # Optional:
    # A = torch.compile(A)

    # Run
    print(A)
    y = A(x)

    # Also should run!
    print(A.H)
    x2 = A.H(y)

    # Also also should run!!
    # x3 = A.N(x)

    # You get the idea
    # y4 = A.fn(x, mps=mps, trj=trj)
    return (
        A,
        B,
        C,
        D,
        F,
        K,
        Nx,
        Ny,
        R,
        S,
        T,
        mps,
        num_interleaves,
        trj,
        x,
        x2,
        x_dims,
        y,
    )


app._unparsable_cell(
    r"""
    Phi = Dense(phi)
    D = Diagonal(dcf)
    T = ImplicitGROGToepNUFFT(trj, inner=(Phi.H @ D @ Phi)
    A = S.H @ T @ Sj

    """,
    name="__"
)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

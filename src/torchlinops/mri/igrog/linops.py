from typing import Optional, Tuple

import torch

from ...core.base import NamedLinop

def ravel(x, shape, dim):
    """
    x: torch.LongTensor, arbitrary shape,
    shape: Shape of the array that x indexes into
    dim: dimension of x that is the "indexing" dimension

    Returns:
    torch.LongTensor of same shape as x but with indexing dimension removed
    """
    out = 0
    for s, i in zip(shape[1:], range(x.shape[dim]-1)):
        out = s * (out + torch.select(x, dim, i))
    out += torch.select(x, dim, -1)
    return out

def multi_grid(x: torch.Tensor, idx: torch.Tensor, final_size: Tuple, raveled: bool = False):
    """Grid values in x to im_size with indices given in idx
    x: [N... I...]
    idx: [I... ndims] or [I...] if raveled=True
    raveled: Whether the idx still needs to be raveled or not

    Returns:
    Tensor with shape [N... final_size]

    Notes:
    Adjoint of multi_index
    """
    if not raveled:
        assert len(final_size) == idx.shape[-1], f'final_size should be of dimension {idx.shape[-1]}'
        idx = ravel(idx, final_size, dim=-1)
    ndims = len(idx.shape)
    assert x.shape[-ndims:] == idx.shape, f'x and idx should correspond in last {ndims} dimensions'
    x_flat = torch.flatten(x, start_dim=-ndims, end_dim=-1) # [N... (I...)]
    idx_flat = torch.flatten(idx)

    batch_dims = x_flat.shape[:-1]
    y = torch.zeros((*batch_dims, *final_size), dtype=x_flat.dtype, device=x_flat.device)
    y = y.reshape((*batch_dims, -1))
    y = y.index_add_(-1, idx_flat, x_flat)
    y = y.reshape(*batch_dims, *final_size)
    return y

def sp_fft(x, dim=None):
    """Matches Sigpy's fft, but in torch"""
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.fftn(x, dim=dim, norm='ortho')
    x = torch.fft.fftshift(x, dim=dim)
    return x

def sp_ifft(x, dim=None, norm=None):
    """Matches Sigpy's fft adjoint, but in torch"""
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.ifftn(x, dim=dim, norm='ortho')
    x = torch.fft.fftshift(x, dim=dim)
    return x


class GriddedNUFFT(NamedLinop):
    def __init__(
            self,
            trj_grd: torch.Tensor,
            im_size: Tuple,
            in_batch_shape: Optional[Tuple] = None,
            out_batch_shape: Optional[Tuple] = None,
            readout_dim: str = 'K',
    ):
        """
        img (input) [A... [C] Nx Ny [Nz]]
        trj_grd: [B... K D] integer-valued tensor in [-N//2, N//2]
        in_batch_shape: Extra dimensions in front of the image, not including spatial dims (e.g. subspace/trs)
        out_batch_shape: Extra dimensions in front of the trajectory, not including coils (e.g. interleaves)
        """

        self.in_batch_shape = in_batch_shape if in_batch_shape is not None else tuple()
        self.out_batch_shape = out_batch_shape if out_batch_shape is not None else tuple()
        ishape = self.in_batch_shape + get2dor3d(im_size)
        oshape = self.in_batch_shape + self.out_batch_shape + (readout_dim,)
        super().__init__(ishape, oshape)
        self.trj = nn.Parameter(trj_grd, requires_grad=False)
        self.readout_dim = readout_dim
        self.im_size = im_size
        self.D = len(im_size)
        self.fft_dim = list(range(-self.D, 0))

    def forward(self, x: torch.Tensor):
        return self.fn(x, self.trj)

    def fn(self, x, /, trj):
        """
        x: [A... Nx Ny [Nz]] # A... may include coils
        trj: [B... K D] integer-valued index tensor
        output: [A... B... K]
        """
        # FFT
        Fx = torch.fft.fftn(x, dim=self.fft_dim, norm='ortho')

        # Index
        A = x.shape[:-self.D]
        omega = (slice(None),) * len(A) + self.trj
        return Fx[omega]

    def adj_fn(self, y, /, trj):
        """
        y: [A... B... K]
        trj: [B... K D] integer-valued index tensor
        output: [A... Nx Ny [Nz]]
        """
        # Un-Index (i.e. grid)
        Fx = multi_grid(y, self.trj, final_size=self.im_size)

        # IFFT
        x = torch.fft.ifftn(x, dim=self.fft_dim, norm='ortho')
        return x

    def split_forward(self, ibatch, obatch):
        return type(self)(
            self.split_forward_fn(ibatch, obatch, self.trj),
            im_size=self.im_size,
            in_batch_shape=self.in_batch_shape,
            out_batch_shape=self.out_batch_shape,
            readout_dim=self.readout_dim,
        )

    def split_forward_fn(self, ibatch, obatch, /, trj):
        """Return data"""
        B_slc = obatch[len(self.in_batch_shape):]
        trj_slc = B_slc + [slice(None)]
        return trj[trj_slc]

    def size(self, dim: str):
        """Get the size of a particular dim, or return
        None if this linop doesn't determine the size
        """
        return self.size_fn(dim, self.trj)

    def size_fn(self, dim: str, /, trj):
        """Functional version of size. Determines sizes from kwargs
        kwargs should be the same as the inputs to fn or adj_fn
        Return None if this linop doesn't determine the size of dim.
        """
        trj_oshapes = self.oshape[len(self.in_batch_shape):] # [B... K]
        if dim in trj_oshapes:
            i = trj_oshapes.index(dim)
            return trj.shape[i]
        return None

from typing import Optional

import torch

from torchlinops.utils import end_pad_with_zeros
from torchlinops._core._linops import NamedLinop


def timeseg(F: NamedLinop, D: Optional[NamedLinop] = None, num_segments: int = 1):
    """
    Convert a NUFFT-style linop to a segmented linop

    Parameters
    ----------
    num_segments : int
        The number of time segments to split the trajectory into
    F : NamedLinop
        NUFFT or GriddedNUFFT linop with a .trj parameter of shape
        [..., readout_dim, spatial_dim]
    D : NamedLinop
        Optional DCF linop with .weight parameter of shape [..., readout_dim]

    Returns
    -------
    last_segment_size
        The size of the last segment (may be shorter than the other segments)
    F_seg : NamedLinop
    [D_seg] : NamedLinop
        The segmented linops with the segment dimension out front:
        F_seg.trj : [num_segments, ..., readout_dim // num_segments, spatial dim]
        D_seg.weight : [num_segments, ..., readout_dim // num_segments]



    """

    def segment_helper(t, num_segments, dim):
        segments = t.chunk(num_segments, dim=dim)
        first_segment = segments[0]
        last_segment = segments[-1]
        last_segment_size = last_segment.shape[dim]
        # Pad last segment
        last_segment = end_pad_with_zeros(
            last_segment, dim, first_segment.shape[dim] - last_segment.shape[dim]
        )
        segments[-1] = last_segment
        return torch.stack(segments, dim=0), last_segment_size

    trj, last_segment_size = segment_helper(F.trj, num_segments, dim=-2)

    F
    if D is not None:
        weight, _ = segment_helper(D.weight, num_segments, dim=-1)
        return last_segment_size, F, D
    return last_segment_size, F


# def _time_segment_reshaper(self,
#                             trj: torch.Tensor,
#                             dcf: torch.Tensor,
#                             nseg: int) -> Union[torch.Tensor, torch.Tensor]:
#     """
#     Helper funciton to process the trajectory and dcf to be time segmented

#     Parameters:
#     -----------
#     trj : torch.tensor <float>
#         The k-space trajectory with shape (nro, npe, ntr, d).
#             we assume that trj values are in [-n/2, n/2] (for nxn grid)
#     dcf : torch.tensor <float>
#         the density comp. functon with shape (nro, npe, ntr)
#     nseg: int
#         number of segments for time segmented model

#     Returns
#     ----------
#     trj : torch.tensor <float32>
#         The segmented k-space trajectory with shape (nseg, nro_new, npe, ntr, d).
#     dcf_rs : torch.tensor <float>
#         The dcf with shape (nseg, nro_new, npe, ntr)
#     edge_segment_size : int
#         true number of readout points for the final segment
#     """

#     # Reshape trj and dcf to support segments
#     split_size = round(trj.shape[0]/nseg)
#     trj_split = torch.split(trj, split_size, dim=0)
#     dcf_split = torch.split(dcf, split_size, dim=0)
#     nseg = len(trj_split)

#     # Move split components into zero-padded tensors
#     trj_rs = torch.zeros((nseg, split_size, *trj.shape[1:]), dtype=torch.float32).to(self.torch_dev)
#     dcf_rs = torch.zeros((nseg, split_size, *dcf.shape[1:]), dtype=torch.float32).to(self.torch_dev)
#     for i in range(nseg):
#         split_size_i = trj_split[i].shape[0]
#         trj_rs[i, :split_size_i] = trj_split[i]
#         dcf_rs[i, :split_size_i] = dcf_split[i]
#     edge_segment_size = trj_split[-1].shape[0]

#     return trj_rs, dcf_rs, edge_segment_size

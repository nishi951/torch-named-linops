"""Interface to optimized-mrf"""
import torch

from optim_mrf.epg_torch import GREParams, SteadyStateMRF, SequenceWrapper
from optim_mrf.initialization import tgas_load

@dataclass
class SequenceConfig:
    sequence_len: int
    TR: float
    """Repetition time between acquisitions"""
    TI: float
    """Inversion time; For inversion recovery"""
    inv_angle: float
    """In degrees; For inversion recovery"""

def load_sequence(opt):
    # Create sequence (customize this)
    flip_angle, TR, TE = tgas_load()
    flip_angle = flip_angle * 90/75 # ??? but it's in the code
    TE = TE/2 #??? but it's in the code
    fispparams = GREParams(
        flip_angle=flip_angle,
        flip_angle_requires_grad=False,
        TR=opt.init_TR*torch.ones(opt.sequence_len),
        TR_requires_grad=False,
        TE=TE,
        TE_requires_grad=False,
    )
    inv_rec_params = {
        'TI': torch.tensor(opt.init_TI),
        'TI_requires_grad': False,
        'inv_angle': torch.tensor(opt.init_inv_angle * np.pi/180.),
        'spoiler': True,
    }
    seq = SteadyStateMRF(
        fispparams,
        inv_rec_params,
        wait_time=torch.tensor(opt.wait_time)
    )
    seq = SequenceWrapper(seq, opt.nstates, opt.real_signal)
    seq.to(device)
    return seq

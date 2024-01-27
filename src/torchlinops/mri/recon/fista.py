import copy
from dataclasses import dataclass
import gc
from typing import Optional, Callable
import logging

from easydict import EasyDict
import torch
import torch.nn as nn
from tqdm import tqdm

from metrics import l2_grad_norm, l2_loss, ptol
from recon.transforms import abs_quantile_normalize
from utils import Timer

__all__ = ['FISTA', 'FISTAHparams']

logger = logging.getLogger(__name__)

def tensor_dict_to(tensordict, device):
    tensordict = EasyDict({(k, v.to(device))
                           if isinstance(v, torch.Tensor)
                           else (k, v)
                           for k, v in tensordict.items()})
    return tensordict


@dataclass
class FISTAHparams:
    lr: float
    """Learning rate for least-squares gradient descent step"""
    num_iters: int
    """Number of FISTA iterations"""
    log_every: int = 1
    """Number of iterations between logging"""
    state_every: int = 1
    """Number of iterations between logging full state (memory intensive)"""

class FISTA(nn.Module):
    """Functional version of FISTA"""
    def __init__(
            self,
            A: Callable,
            AH: Callable,
            prox: Callable,
            hparams: FISTAHparams,
            AHA: Optional[Callable] = None,
            precond: Optional[Callable] = None,
    ):
        super().__init__()
        self.A = A
        self.AH = AH
        self.prox = prox
        self.hparams = hparams
        self.precond = precond if precond is not None else lambda x: x

        # Can provide an optimized operator here instead
        if AHA is not None:
            logger.debug('Using user-supplied AHA')
            self.AHA = AHA
        else:
            self.AHA = lambda x: AH(A(x))

        self.states = []
        self.logs = []

    def run(self, b=None, init=None, AHb=None):
        """
        b: Input data
        init: Optional start
        AHb: Optional precomputed AHb (for grad step)
        """
        assert (b is not None) or (AHb is not None), 'At least one of (b, AHb) should not be provided'
        if init is None:
            logger.debug('Initialized with adjoint')
            if AHb is not None:
                x = AHb.clone()
            elif b is not None:
                x = self.AH(b)
        else:
            logger.debug('Initialized from argument')
            x = init
        if AHb is None:
            logger.debug('Precomputing AHb')
            AHb = self.AH(b)

        logger.debug('>> Starting...')
        if len(self.states) > 0:
            logger.warning('Restarting FISTA and overwriting logs.')
        self.states = []
        self.logs = []
        timer = Timer(name="fista iter", log_level=logging.DEBUG)
        s = EasyDict({
            'x': x.clone(),
            'z': x.clone(),
        })
        self.states.append((-1, tensor_dict_to(copy.deepcopy(s), 'cpu')))
        oldstate = copy.deepcopy(s)
        self.log_all(iteration=-1, x=s.x, x_old=s.x, b=b, AHb=AHb, gr=None)

        for k in tqdm(
                range(self.hparams.num_iters),
                desc='FISTA',
        ):
            logger.debug(f'>>> Starting iteration {k:03d}... ')
            torch.cuda.empty_cache()
            gc.collect()
            with timer:
                s.x = s.z.clone()
                # logger.info(f'Norm of x(before): {torch.linalg.norm(s.x)}')
                # tmp = self.AHA(s.x)
                # logger.info(f'Norm of x(after): {torch.linalg.norm(tmp)}')
                gr = self.AHA(s.x) - AHb
                pgr = self.precond(gr)
                #logger.info(torch.linalg.norm(gr).item())
                s.x = s.x - self.hparams.lr*pgr
                s.x = self.prox(s.x)
                # Monitor magnitude of x
                # Apply acceleration
                step  = k/(k + 3)
                s.z = s.x + step * (s.x - oldstate.x)
            logger.debug(f'{timer.total:0.4f} s')
            oldstate = copy.deepcopy(s)

            # Logging
            if not (k % self.hparams.log_every):
                self.log_all(k, s.x, oldstate.x, b, AHb, gr)
            if not (k % self.hparams.state_every):
                save_state = tensor_dict_to(copy.deepcopy(oldstate), 'cpu')
                self.states.append((k, save_state))
        return s

    def log_all(self, iteration, x, x_old, b, AHb, gr):
        log = EasyDict()
        log.ptol = ptol(x_old, x).item()
        log.data_loss = l2_loss(x, b, self.A, self.AH, self.AHA, AHb).item()
        log.grad_norm = l2_grad_norm(gr, x, b, self.A, self.AH, self.AHA, AHb).item()
        self.logs.append((iteration, log))
        return log



def FISTA_fn(A, AH, b, proxop, num_iters, lr=1., init=None, AHb=None):
    """FISTA algorithm with fixed step size
    in pytorch (functional form)
    can optionally specify initialization
    can specify AHb if it's already computed
    """
    AHb = AHb if AHb is not None else AH.apply(b)

    if init is None:
        print(">> Starting reconstruction from AHb.", flush=True)
        x = AHb.clone().detach()
    else:
        print(">> Starting reconstruction from init.", flush=True)
        x = init.clone().detach()
    z = x.clone()
    ptol = None

    if num_iters <= 0:
        return (x, ptol)

    print(">> Starting iterations: ", flush=True)
    log = []
    for k in tqdm(range(num_iters), total=num_iters):
        #print(">>> Starting iteration %03d... " % k, end="", flush=True)
        start_time = time.perf_counter()
        x_old = x.clone()
        x     = z.clone()

        gr    = AH.apply(A.apply(x)) - AHb
        x     = proxop.apply(1, x - lr*gr)
        if k == 0:
            z = x
        else:
            step  = k/(k + 3)
            z     = x + step * (x - x_old)
        end_time = time.perf_counter()
        # print("done. Time taken: %0.2f seconds." % (end_time - start_time),
        #       flush=True)
        res = EasyDict({
            'x': x.clone().detach(),
            'z': z.clone().detach(),
        })
        log.append(struct2np(res))
    with torch.no_grad():
        ptol = 100 * torch.linalg.norm(x_old - x)/torch.linalg.norm(x)
        print(">> Iterative tolerance percentage achieved: %0.2f" % ptol.item(),
              flush=True)
    return (log, ptol)

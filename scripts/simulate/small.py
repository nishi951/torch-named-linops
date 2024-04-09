from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

import sigpy as sp
import sigpy.mri as mri

from torchlinops.utils import Experiment, get_device, struct2np
from torchlinops.mrf.simulator import SimulatorConfig, load_sequence, SequenceConfig

from optim_mrf.initialize import tgas_load


@dataclass
class Config:
    input_dir: Path
    output_dir: Path
    output_name: str = "small.npz"
    im_size: Tuple[int, int] = (64, 64)
    sequence: SequenceConfig = field(
        default_factory=lambda: SequenceConfig(
            nstates=100,
            real_signal=True,
        )
    )
    log_level: int = logging.INFO
    device_idx: int = -1
    # reload_cache: bool = False


def main(opt: Config):
    simulated_data = generate_simulated_data(opt)
    np.savez(opt.output_dir / opt.output_name, data=simulated_data)


def generate_simulated_data(opt: Config):
    device = get_device(opt.device_idx)

    # Load MRF sequence and generate dictionary
    seq = load_sequence(opt.sequence, device)
    signal_dict = ...
    phi = ...

    # Load phantom and maps
    img = quantitative_shepp_logan(opt.im_size)
    mps = mri.birdcage_maps(opt.im_size)

    # Load trajectory and dcf
    trj = ...
    dcf = mri.pipe_menon_dcf(trj, img_shape=im_size, max_iter=30)

    simulator = Simulator(
        seq,
        trj,
        opt.im_size,
        params=opt.simulator,
        device_idx=opt.device_idx,
    )
    ksp = simulator.simulate(img, mps)
    out = {
        "img": img,
        "ksp": ksp,
        "mps": mps,
        "trj": trj,
        "dcf": dcf,
        "phi": phi,
    }
    return struct2np(out)


if __name__ == "__main__":
    exp = Experiment(Config, main)
    exp.run()

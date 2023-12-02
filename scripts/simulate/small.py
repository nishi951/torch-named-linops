from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

from torchlinops.utils import Experiment, get_device
from torchlinops.mrf.simulator import SimulatorConfig, load_sequence, SequenceConfig

from optim_mrf.initialize import tgas_load


@dataclass
class Config:
    input_dir: Path
    output_dir: Path
    sequence: SequenceConfig = field(default_factory=lambda: SequenceConfig(
        nstates=100,
        real_signal=True,
    ))
    log_level: int = logging.INFO
    device_idx: int = -1
    #reload_cache: bool = False

def main(opt: Config):
    device = get_device(opt.device_idx)
    seq = load_sequence(opt.sequence, device)

if __name__ == '__main__':
    exp = Experiment(Config, main)
    exp.run()

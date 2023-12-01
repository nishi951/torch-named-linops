from dataclasses import dataclass, field, asdict

from torchlinops.utils import Experiment
from torchlinops.mrf.simulator import SimulatorConfig

@dataclass
class Config:
    input_dir: Path
    output_dir: Path


def main(opt: Config):
    #

if __name__ == '__main__':
    exp = Experiment(Config, main)
    exp.run()

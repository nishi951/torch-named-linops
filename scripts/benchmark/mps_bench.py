import torch
import tyro


@dataclass
class Config:
    num_reps: int = 10
    """Number of repetitions"""
    num_coils: int = 12
    """Number of coils"""
    mat_size: Tuple[int] = 100
    """Matrix size to use"""


def main(opt: Config):
    x = torch.randn(mat_size, mat_size, dtype=torch.complex64)
    S = torch.randn(num_coils, mat_size, mat_size, dtype=torch.complex64)




if __name__ == '__main__':
    opt = tyro.cli(Config)
    main(opt)

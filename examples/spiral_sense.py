import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from torchlinops.mri.sim.spiral2d import (
    Spiral2dSimulator,
    Spiral2dSimulatorConfig,
)
from torchlinops import Batch

from benchmark.torch_benchmark import TorchHandler

config = Spiral2dSimulatorConfig(
    im_size=(128, 128),
    num_coils=20,
    noise_std=0.,
    nufft_backend='fi',
    spiral_2d_kwargs={
        "n_shots": 16,
        "alpha": 1.5,
        "f_sampling": 0.4,
        "g_max": 0.04,
        "s_max": 100.0,
    }
)
simulator = Spiral2dSimulator(config)
data = simulator.data
device = torch.device('cuda:0')
batching = False
toeplitz = True
A = simulator.make_linop(data.trj, data.mps, toeplitz)
#A.to(device)

if batching:
    AN = Batch(
        A.N.to(device),
        input_device=device,
        output_device=device,
        input_dtype=torch.complex64,
        output_dtype=torch.complex64,
        name='AN',
        C=1,
    )
else:
    AN = A.N.to(device)

# Check speed and memory usage
x = data.img.to(device)
tbh = TorchHandler()
tbh.bench_start()
for i in range(10):
    tbh.trial_start(None, i)
    out = AN(x)
    tbh.trial_end(None, i)
tbh.bench_end()

print('batching: ', batching)
print('toeplitz: ', toeplitz)
print(f'peak memory: {tbh.result.max_mem_bytes//1000:0.1f} Kb')
print(f'avg runtime: {np.mean(tbh.result.timings_ms):0.2f} ms')

matplotlib.use('WebAgg')
plt.figure()
plt.imshow(np.abs(x.detach().cpu().numpy()))
plt.title('Input')
plt.figure()
plt.imshow(np.abs(out.detach().cpu().numpy()))
plt.title('A.N(input)')
plt.show()

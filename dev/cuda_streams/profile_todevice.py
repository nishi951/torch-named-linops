import torch
import torch.profiler
from torch.cuda import Event

from torchlinops import Dense, Dim
from torchlinops.linops.device import DeviceSpec, ToDevice

assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs"

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

N = 8192
W = torch.randn(N, N, device=device0)
x = torch.randn(N, device=device0)

linop1 = Dense(W, Dim("MN"), ishape=Dim("N"), oshape=Dim("M")).to(device0)
linop2 = Dense(W, Dim("MN"), ishape=Dim("N"), oshape=Dim("M")).to(device0)

y1_ready = Event()
y2_ready = Event()

y1_to_gpu1 = ToDevice(
    DeviceSpec(device0),
    DeviceSpec(device1),
    ioshape=Dim("M"),
    input_ready_event=y1_ready,
)

y2_to_gpu1 = ToDevice(
    DeviceSpec(device0),
    DeviceSpec(device1),
    ioshape=Dim("M"),
    input_ready_event=y2_ready,
)

fromdevice_to_cpu = ToDevice(
    DeviceSpec(device1),
    DeviceSpec(device0),
    ioshape=Dim("M"),
)

compute_linop = Dense(W, Dim("MN"), ishape=Dim("N"), oshape=Dim("M")).to(device1)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=False,
    with_stack=False,
    profile_memory=False,
) as prof:
    y1 = linop1(x)
    y1_ready.record()
    y2 = linop2(x)
    y2_ready.record()

    x1 = y1_to_gpu1(y1)
    x2 = y2_to_gpu1(y2)

    z1 = compute_linop(x1)
    z2 = compute_linop(x2)

    out1 = fromdevice_to_cpu(z1)
    out2 = fromdevice_to_cpu(z2)

    torch.cuda.synchronize(device0)
    torch.cuda.synchronize(device1)

trace_file = "todevice_trace.json"
prof.export_chrome_trace(trace_file)

print(f"Chrome trace written to: {trace_file}")
print("Open chrome://tracing and load the file.")

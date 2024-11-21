import torch

from torchlinops import Dense, Batch
from torch.profiler import profile, record_function, ProfilerActivity


def main():
    device_idxs = [0, 1]
    devices = [torch.device(f"cuda:{idx}") for idx in device_idxs]
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B, M, N = 6, 100000, 10000
    # Set up
    weight = torch.randn(B, M, N, device=devices[0])
    A = Dense(weight, ("B", "M", "N"), ishape=("N",), oshape=("B", "M"))
    A.to(devices[0])
    A_batch = Batch(A, devices[0], devices[0], torch.float32, torch.float32, B=1)
    # Prepare streams
    # streams = [torch.cuda.Stream(devices[d % len(devices)]) for d in range(B)]
    default_streams = [torch.cuda.Stream(device) for device in devices]
    streams = [default_streams[d % len(devices)] for d in range(B)]
    for linop, stream in zip(A_batch._linops, streams):
        linop.to(stream.device)

    y = torch.empty(B, M, device=devices[0])
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=lambda p: p.export_chrome_trace(f"./trace.json"),
    ) as prof:
        for i in range(10):
            for j, s in enumerate(streams):
                with torch.cuda.stream(s):
                    outputs = A_batch._linops[j](torch.randn(N, device=s.device))
        for j, output in enumerate(outputs):
            y[j] = output.to(y.device)


if __name__ == "__main__":
    main()

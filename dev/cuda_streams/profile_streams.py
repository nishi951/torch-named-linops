import torch

from torchlinops import Dense, Batch
from torch.profiler import profile, record_function, ProfilerActivity


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B, M, N = 3, 10000, 1000
    # Set up
    weight = torch.randn(B, M, N, device=device)
    A = Dense(weight, ("B", "M", "N"), ishape=("N",), oshape=("B", "M"))
    A.to(device)
    A_batch = Batch(A, device, device, torch.float32, torch.float32, B=1)
    A_batch = torch.compile(A_batch)

    x = torch.randn(N, device=device)
    y = torch.empty(B, M, device=device)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=lambda p: p.export_chrome_trace(f"./trace.json"),
    ) as prof:
        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        for i in range(10):
            with torch.cuda.stream(s0):
                y[0] = A_batch._linops[0](x)
            with torch.cuda.stream(s1):
                y[1] = A_batch._linops[1](x)
            with torch.cuda.stream(s2):
                y[2] = A_batch._linops[2](x)


if __name__ == "__main__":
    main()

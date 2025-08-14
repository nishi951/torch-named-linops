import time

import torch
from torch.cuda import Stream
from torch.profiler import ProfilerActivity, profile, record_function

from torchlinops import BatchSpec, Dense, Dim, create_batched_linop


def main():
    device_idxs = [0, 1]
    devices = [torch.device(f"cuda:{idx}") for idx in device_idxs]
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B, M, N = 6, 100000, 10000
    # Set up
    weight = torch.randn(B, M, N, device=devices[0])
    As = {}
    for device in devices:
        As[device] = Dense(weight, Dim("BMN"), ishape=Dim("BN"), oshape=Dim("BM")).to(
            device
        )

    # A.to(devices[0])
    # A = create_batched_linop(
    #     A, BatchSpec({"B": 3}, device_matrix=devices, base_device=devices[0])
    # )

    # Prepare streams
    streams = {device: Stream(device) for device in devices}

    x = torch.randn(B, N, device=devices[0])
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=lambda p: p.export_chrome_trace("./trace.json"),
    ) as prof:
        for i in range(10):
            for device in devices:
                stream = streams[device]
                with torch.cuda.stream(stream):
                    x_dev = x.to(device, non_blocking=True)
                    y = As[device](x_dev)
            time.sleep(0.2)
        for device in devices:
            torch.cuda.synchronize(device)


if __name__ == "__main__":
    main()

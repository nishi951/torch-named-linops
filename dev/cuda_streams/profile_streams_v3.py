import time
import logging

import torch
from torch.cuda import Stream
from torch.profiler import ProfilerActivity, profile, record_function

from torchlinops import BatchSpec, Dense, Dim, ToDevice, create_batched_linop
from torchlinops.utils import setup_console_logger

logger = logging.getLogger("torchlinops-dev")


def main(todevice: bool = False):
    device_idxs = [0, 1]
    devices = [torch.device(f"cuda:{idx}") for idx in device_idxs]
    streams = {device: Stream(device) for device in devices}
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B, M, N = 6, 100000, 10000
    # Set up
    weight = torch.randn(B, M, N, device=devices[0])
    As = {}
    base_device = devices[0]
    for device in devices:
        linop = Dense(weight, Dim("BMN"), ishape=Dim("BN"), oshape=Dim("BM")).to(device)

        if todevice:
            base_stream = torch.cuda.default_stream(base_device)
            transfer_stream = streams[base_device]
            target_stream = torch.cuda.default_stream(device)
            linop.stream = target_stream
            As[device] = (
                ToDevice(
                    idevice=device,
                    odevice=base_device,
                    ioshape=Dim("BM"),
                    istream=target_stream,
                    ostream=base_stream,
                )
                @ linop
                @ ToDevice(
                    idevice=base_device,
                    odevice=device,
                    ioshape=Dim("BN"),
                    istream=transfer_stream,
                    ostream=target_stream,
                )
            )
        else:
            As[device] = linop

    # A.to(devices[0])
    # A = create_batched_linop(
    #     A, BatchSpec({"B": 3}, device_matrix=devices, base_device=devices[0])
    # )

    x = torch.randn(B, N, device=base_device)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=lambda p: p.export_chrome_trace("./trace.json"),
    ) as prof:
        if todevice:
            logger.info("Running linop-based stream/device management")
            for i in range(10):
                for device in devices:
                    y = As[device](x)
            for device in devices:
                torch.cuda.synchronize(device)
        else:
            logger.info("Running manual stream/device management")
            # Manual
            # - Want computation to occur on default stream of each device
            # - In Pytorch, the Transfer is always registered on a stream of the SOURCE device
            # - On the base device (where data starts and ends), transfer must occur on separate stream.
            # - Transfer stream should not wait on the computation stream
            for i in range(10):
                base_stream = torch.cuda.default_stream(base_device)
                for device in devices:
                    target_stream = torch.cuda.default_stream(device)
                    transfer_stream = streams[base_device]
                    # ToDevice
                    with torch.cuda.stream(transfer_stream):
                        x2 = x.to(device, non_blocking=True)
                    # Don't mess with x until transfer stream is done
                    x.record_stream(transfer_stream)
                    # Target should wait for transfer to complete before starting work
                    target_stream.wait_stream(transfer_stream)

                    # Linop
                    with torch.cuda.stream(target_stream):
                        x3 = As[device](x2)
                    # Necessary if target_stream is not a default stream
                    x2.record_stream(target_stream)

                    # ToDevice
                    # Using target stream here forces wait until linop computation is done.
                    # Also remember that transfer is forced to be on a source device stream.
                    with torch.cuda.stream(target_stream):
                        out = x3.to(base_device, non_blocking=True)
                    # Don't mess with x3 until transfer is done (may be unnecessary)
                    x3.record_stream(target_stream)
                    # Transfer stream should not wait for the target stream to complete the transfer
                    # transfer_stream.wait_stream(target_stream) # NO

                    # Instead, the base stream should wait
                    base_stream.wait_stream(target_stream)

                # time.sleep(0.2)
            for device in devices:
                torch.cuda.synchronize(device)


if __name__ == "__main__":
    setup_console_logger()
    main(todevice=True)

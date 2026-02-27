import torch
import torch.profiler
import time

assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs"

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

# Create a dedicated transfer stream on GPU0
transfer_stream = torch.cuda.Stream(device=device0)

# Large tensor to make compute + copy visible
N = 8192
y = torch.randn(N, N, device=device0)


def slow1(y):
    # heavy matmul chain to keep GPU busy
    x = y
    for _ in range(5):
        x = x @ y
    return x


def slow2(y):
    z = y
    for _ in range(5):
        z = z @ y
    return z


with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=False,
    with_stack=False,
    profile_memory=False,
) as prof:
    # --- 1. First slow function (produces x) ---
    x = slow1(y)

    # Record fine-grained event right after slow1
    event = torch.cuda.Event()
    event.record()  # recorded on default stream (cuda:0)

    # --- 2. Second slow function (independent of x) ---
    z = slow2(y)

    # --- 3. Transfer on separate stream ---
    with torch.cuda.stream(transfer_stream):
        transfer_stream.wait_event(event)
        x1 = x.to(device1, non_blocking=True)

    # Make sure everything finishes before profiler exits
    torch.cuda.synchronize(device0)
    torch.cuda.synchronize(device1)

# Export Chrome trace
trace_file = "multi_stream_trace.json"
prof.export_chrome_trace(trace_file)

print(f"Chrome trace written to: {trace_file}")
print("Open chrome://tracing and load the file.")

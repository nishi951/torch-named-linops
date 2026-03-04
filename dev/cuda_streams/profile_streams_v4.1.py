import torch
import torch.profiler

assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs"

device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

transfer_stream = torch.cuda.Stream(device=device0)

N = 8192
y = torch.randn(N, N, device=device0)


def slow1(y):
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
    # --- 1. First slow function ---
    x = slow1(y)

    # --- 2. Immediately enqueue second slow function ---
    z = slow2(y)

    # --- 3. Transfer on separate stream (no event) ---
    with torch.cuda.stream(transfer_stream):
        x1 = x.to(device1, non_blocking=True)

    torch.cuda.synchronize(device0)
    torch.cuda.synchronize(device1)

trace_file = "multi_stream_no_event_trace.json"
prof.export_chrome_trace(trace_file)

print(f"Chrome trace written to: {trace_file}")
print("Open chrome://tracing and load the file.")

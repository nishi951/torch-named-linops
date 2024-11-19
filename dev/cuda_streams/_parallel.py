import json

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity

try:
    mp.set_start_method("forkserver")
except RuntimeError:
    pass

__all__ = ["parallel_call", "merge_traces"]


def parallel_call(
    modules: list[nn.Module],
    args: list[tuple],
    do_profile: bool = False,
):
    """Run a list of modules on a list of arguments in parallel

    Note: if the modules have any parameters that require gradient, this might not work.

    """
    if len(modules) != len(args):
        raise ValueError(
            f"modules and args must be same length but got {len(modules) } != {len(args)}"
        )
    # Initialize input queues, output_queues, and received queues
    n_procs = len(modules)
    qins = [mp.Queue() for _ in range(n_procs)]
    qouts = [mp.Queue() for _ in range(n_procs)]
    recvs = [mp.Event() for _ in range(n_procs)]

    # Start processes and send data
    procs = []
    for rank, mod, args, qin, qout, recv in zip(
        range(n_procs), modules, args, qins, qouts, recvs
    ):
        proc = mp.Process(target=call_module, args=(rank, qin, qout, recv, do_profile))
        proc.start()
        qin.put((mod, args))
        procs.append(proc)

    # Receive data, signal child processes to exit,and join processes
    outputs = []
    for proc, qout, recv in zip(procs, qouts, recvs):
        outputs.append(qout.get())
        recv.set()
        proc.join()

    return outputs


def call_module(
    rank: int,
    qin: mp.Queue,
    qout: mp.Queue,
    recv: mp.Event,
    do_profile: bool = False,
):
    """Helper function for parallel call

    rank : int
        The id of this process

    """
    # Get module and args from input queue
    mod, args = qin.get()

    # Run module on args
    if do_profile:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=lambda p: p.export_chrome_trace(
                f"./worker_{rank}_trace.json"
            ),
        ) as prof:
            out = mod(*args)
    else:
        out = mod(*args)

    # Put output in the output queue
    qout.put(out)

    # Wait for parent process to signal received
    recv.wait()


def merge_traces(input_files, output_file):
    """

    Example usage
    trace_files = glob.glob("./worker_*_trace.json")
    merge_traces(trace_files, "merged_trace.json")
    """
    all_traces = {}
    all_traces["traceEvents"] = []

    for i, trace_file in enumerate(input_files):
        with open(trace_file, "r") as f:
            trace_data = json.load(f)
            events = trace_data["traceEvents"]
            for event in events:
                event["pid"] = str(event["pid"]) + f"_{i}"

            # Important step
            all_traces["traceEvents"].extend(events)
            # Less important steps
            all_traces["schemaVersion"] = trace_data["schemaVersion"]
            all_traces["deviceProperties"] = trace_data["deviceProperties"]
    all_traces["traceName"] = str(output_file)

    # Write combined traces to a new file
    with open(output_file, "w") as f:
        json.dump(all_traces, f)


### Testing ###
class TestAdd(nn.Module):
    """Needs to be defined at top level of a module"""

    def __init__(self, bias):
        super().__init__()
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, a):
        def flipsign(x):
            """Test functions defined within methods"""
            return -x

        for i in range(1000):
            flipsign(a + self.bias)
        return flipsign(a + self.bias)


if __name__ == "__main__":
    n_parallel = 3
    devs = [torch.device(f"cuda:{i}") for i in range(n_parallel)]
    modules = [
        TestAdd(torch.randn(i + 1, 1, device=devs[i])) for i in range(n_parallel)
    ]
    args = [(torch.randn(i + 1, device=devs[i]),) for i in range(n_parallel)]
    outputs = parallel_call(modules, args, do_profile=True)
    print([f"{t.shape} : {t.device}" for t in outputs])
    import glob, json

    trace_files = glob.glob("./worker_*_trace.json")
    print(trace_files)
    merge_traces(trace_files, "merged_trace.json")

    breakpoint()

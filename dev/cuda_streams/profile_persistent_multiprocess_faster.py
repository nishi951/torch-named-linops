from warnings import warn
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torchlinops import Dense, Batch
from torch.profiler import profile, record_function, ProfilerActivity

try:
    mp.set_start_method("forkserver")
except RuntimeError:
    pass


def main():
    print(f"Found {mp.cpu_count()} cores")
    # Set up
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B, M, N = 5, 100000, 10000
    weight = torch.randn(B, M, N, device=device)
    A = Dense(weight, ("B", "M", "N"), ishape=("N",), oshape=("B", "M"))
    A.to(device)
    A_batch = Batch(A, device, device, torch.float32, torch.float32, B=1)

    # Initialize pool
    pool = PersistentWorkerPool(A_batch._linops, do_profile=True)

    # Allocate buffers
    inputs = [(torch.randn(N, device=device),) for _ in range(B)]
    # Repeat to show warming up
    for i in range(10):
        for j in range(len(inputs)):
            inputs[j][0].data = torch.randn_like(inputs[j][0])
        # Issue a batch of data to the child processes
        for b in range(B):
            outputs = pool.parallel_run(inputs)
    pool.close()


class PersistentWorkerPool:
    def __init__(self, modules: list[nn.Module], do_profile: bool = False):
        self.workers = []
        for i, mod in enumerate(modules):
            self.workers.append(ModuleWorker(mod, rank=i, do_profile=do_profile))

    def __len__(self):
        return len(self.workers)

    def parallel_run(self, args_list: list[tuple]):
        """Run workers with arguments"""
        if len(args_list) != len(self):
            warn(
                f"Persistent Worker Pool called with different numbers of arguments {len(args_list)} than underlying workers: {len(self)}"
            )
        for args, worker in zip(args_list, self.workers):
            worker.call(args)

        outputs = []
        for worker in self.workers:
            outputs.append(worker.get())
        return outputs

    def close(self):
        for worker in self.workers:
            worker.stop()


# Special token for killing the worker

STOP = "STOP"


class ModuleWorker:
    """A torch module inside a process"""

    def __init__(self, mod: nn.Module, rank: int, do_profile):
        self.rank = rank
        self.do_profile = do_profile
        self.qin = mp.SimpleQueue()
        self.qout = mp.SimpleQueue()
        self.reset = mp.Event()
        self.all_done = mp.Event()

        # Initialize worker
        self._worker = mp.Process(
            target=_worker_main,
            args=(rank, self.qin, self.qout, self.reset, self.all_done, do_profile),
        )
        self._worker.start()
        self.qin.put(mod)

    def call(self, args: tuple):
        """Nonblocking"""
        self.qin.put(args)

    def get(self):
        """Blocking"""
        out = self.qout.get()
        self.reset.set()
        return out

    def stop(self):
        self.qin.put(STOP)
        self._worker.join()
        self._worker.close()  # For completeness


def _worker_main(
    rank: int,
    qin: mp.Queue,
    qout: mp.Queue,
    reset: mp.Event,
    all_done: mp.Event,
    do_profile: bool = False,
):
    mod = qin.get()
    if do_profile:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=lambda p: p.export_chrome_trace(
                f"./worker_{rank}_trace.json"
            ),
        ) as prof:
            while True:
                # Main loop
                # Ready state
                reset.clear()
                args = qin.get()
                if args == STOP:
                    break
                # Running state
                out = mod(*args)
                qout.put(out)
                # Resting state
                reset.wait()

    else:
        while True:
            # Main loop
            # Ready state
            args = qin.get()
            if args == STOP:
                break
            # Running state
            out = mod(*args)
            qout.put(out)
            # Resting state
            reset.wait()


if __name__ == "__main__":
    main()

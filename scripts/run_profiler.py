from typing import Callable

import torch

import nerfacc


class Profiler:
    def __init__(self, warmup=10, repeat=1000):
        self.warmup = warmup
        self.repeat = repeat

    def __call__(self, func: Callable):
        # warmup
        for _ in range(self.warmup):
            func()
        torch.cuda.synchronize()

        # profile
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
        ) as prof:
            for _ in range(self.repeat):
                func()
            torch.cuda.synchronize()

        # return
        events = prof.key_averages()
        self_cpu_time_total = (
            sum([event.self_cpu_time_total for event in events]) / self.repeat
        )
        self_cuda_time_total = (
            sum([event.self_cuda_time_total for event in events]) / self.repeat
        )
        self_cuda_memory_usage = max(
            [event.self_cuda_memory_usage for event in events]
        )
        return (
            self_cpu_time_total,  # in us
            self_cuda_time_total,  # in us
            self_cuda_memory_usage,  # in bytes
        )


def main():
    device = "cuda:0"
    torch.manual_seed(42)
    profiler = Profiler(warmup=10, repeat=1000)

    # contract
    print("* contract")
    x = torch.rand([1024, 3], device=device)
    roi = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32, device=device)
    fn = lambda: nerfacc.contract(
        x, roi=roi, type=nerfacc.ContractionType.UN_BOUNDED_TANH
    )
    cpu_t, cuda_t, cuda_bytes = profiler(fn)
    print(f"{cpu_t:.2f} us, {cuda_t:.2f} us, {cuda_bytes / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()

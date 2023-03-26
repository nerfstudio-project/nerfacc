from typing import Callable

import torch
import tqdm

import nerfacc

# timing
# https://github.com/pytorch/pytorch/commit/d2784c233bfc57a1d836d961694bcc8ec4ed45e4


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
        # print(events.table(sort_by="self_cpu_time_total", row_limit=10))
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
    profiler = Profiler(warmup=10, repeat=100)

    # # contract
    # print("* contract")
    # x = torch.rand([1024, 3], device=device)
    # roi = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.float32, device=device)
    # fn = lambda: nerfacc.contract(
    #     x, roi=roi, type=nerfacc.ContractionType.UN_BOUNDED_TANH
    # )
    # cpu_t, cuda_t, cuda_bytes = profiler(fn)
    # print(f"{cpu_t:.2f} us, {cuda_t:.2f} us, {cuda_bytes / 1024 / 1024:.2f} MB")

    # rendering
    print("* rendering")
    batch_size = 81920
    rays_o = torch.rand((batch_size, 3), device=device)
    rays_d = torch.randn((batch_size, 3), device=device)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    ray_indices, t_starts, t_ends = nerfacc._ray_marching(
        rays_o,
        rays_d,
        near_plane=0.1,
        far_plane=1.0,
        render_step_size=1e-1,
    )
    sigmas = torch.randn_like(t_starts, requires_grad=True)
    fn = (
        lambda: nerfacc.render_weight_from_density(
            ray_indices, t_starts, t_ends, sigmas
        )
        .sum()
        .backward()
    )
    fn()
    torch.cuda.synchronize()
    for _ in tqdm.tqdm(range(100)):
        fn()
        torch.cuda.synchronize()

    cpu_t, cuda_t, cuda_bytes = profiler(fn)
    print(f"{cpu_t:.2f} us, {cuda_t:.2f} us, {cuda_bytes / 1024 / 1024:.2f} MB")

    packed_info = nerfacc.pack_info(ray_indices, n_rays=batch_size)
    fn = (
        lambda: nerfacc._vol_rendering._RenderingDensity.apply(
            packed_info, t_starts, t_ends, sigmas, 0
        )
        .sum()
        .backward()
    )
    fn()
    torch.cuda.synchronize()
    for _ in tqdm.tqdm(range(100)):
        fn()
        torch.cuda.synchronize()
    cpu_t, cuda_t, cuda_bytes = profiler(fn)
    print(f"{cpu_t:.2f} us, {cuda_t:.2f} us, {cuda_bytes / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()

"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import glob
import os
from subprocess import DEVNULL, call

from rich.console import Console
from torch.utils.cpp_extension import load

PATH = os.path.dirname(os.path.abspath(__file__))


def cuda_toolkit_available():
    """Check if the nvcc is avaiable on the machine."""
    try:
        call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
        return True
    except FileNotFoundError:
        return False


_C = None
if cuda_toolkit_available():
    console = Console()
    with console.status(
        "[bold yellow]Setting up CUDA (This may take a few minutes the first time)",
        spinner="bouncingBall",
    ):
        _C = load(
            name="nerfacc_cuda",
            sources=glob.glob(os.path.join(PATH, "csrc/*.cu")),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
        )
else:
    console = Console()
    console.print("[bold red]No CUDA toolkit found. NerfAcc will be disabled.")

__all__ = ["_C"]

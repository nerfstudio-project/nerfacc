"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import glob
import os
from subprocess import DEVNULL, call

from rich.console import Console
from torch.utils.cpp_extension import _get_build_directory, load

PATH = os.path.dirname(os.path.abspath(__file__))


def cuda_toolkit_available():
    """Check if the nvcc is avaiable on the machine."""
    try:
        call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
        return True
    except FileNotFoundError:
        return False


def load_extention(name: str):
    return load(
        name=name,
        sources=glob.glob(os.path.join(PATH, "csrc/*.cu")),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
    )


_C = None
name = "nerfacc_cuda"
if os.listdir(_get_build_directory(name, verbose=False)) != []:
    # If the build exists, we assume the extension has been built
    # and we can load it.
    _C = load_extention(name)
else:
    # First time to build the extension
    if cuda_toolkit_available():
        with Console().status(
            "[bold yellow]NerfAcc: Setting up CUDA (This may take a few minutes the first time)",
            spinner="bouncingBall",
        ):
            _C = load_extention(name)
    else:
        Console().print(
            "[yellow]NerfAcc: No CUDA toolkit found. NerfAcc will be disabled.[/yellow]"
        )

__all__ = ["_C"]

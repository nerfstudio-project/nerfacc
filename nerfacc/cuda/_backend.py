"""Setup cuda backend."""
import glob
import os
from subprocess import DEVNULL, call

from torch.utils.cpp_extension import load

PATH = os.path.dirname(os.path.abspath(__file__))


def cuda_toolkit_available():
    """Check if the nvcc is avaiable on the machine."""
    # https://github.com/idiap/fast-transformers/blob/master/setup.py
    try:
        call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
        return True
    except FileNotFoundError:
        return False


if cuda_toolkit_available():
    sources = glob.glob(os.path.join(PATH, "csrc/*.cu"))
else:
    sources = glob.glob(os.path.join(PATH, "csrc/*.cpp"))

extra_cflags = ["-O3"]
extra_cuda_cflags = ["-O3"]
_C = load(
    name="nerfacc_cuda",
    sources=sources,
    extra_cflags=extra_cflags,
    extra_cuda_cflags=extra_cuda_cflags,
)

__all__ = ["_C"]

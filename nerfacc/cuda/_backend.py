"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import glob
import json
import os
import shutil
import urllib.request
import zipfile
from subprocess import DEVNULL, call

from packaging import version
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


def cuda_toolkit_version():
    """Get the cuda toolkit version."""
    cuda_home = os.path.join(os.path.dirname(shutil.which("nvcc")), "..")
    if os.path.exists(os.path.join(cuda_home, "version.txt")):
        with open(os.path.join(cuda_home, "version.txt")) as f:
            cuda_version = f.read().strip().split()[-1]
    elif os.path.exists(os.path.join(cuda_home, "version.json")):
        with open(os.path.join(cuda_home, "version.json")) as f:
            cuda_version = json.load(f)["cuda"]["version"]
    else:
        raise RuntimeError("Cannot find the cuda version.")
    return cuda_version


name = "nerfacc_cuda"
build_dir = _get_build_directory(name, verbose=False)
extra_include_paths = []
extra_cflags = ["-O3"]
extra_cuda_cflags = ["-O3"]

_C = None
if cuda_toolkit_available():
    # # we need cub >= 1.15.0 which is shipped with cuda >= 11.6, so download if
    # # necessary. (compling does not garentee to success)
    # if version.parse(cuda_toolkit_version()) < version.parse("11.6"):
    #     target_path = os.path.join(build_dir, "cub-1.17.0")
    #     if not os.path.exists(target_path):
    #         zip_path, _ = urllib.request.urlretrieve(
    #             "https://github.com/NVIDIA/cub/archive/1.17.0.tar.gz",
    #             os.path.join(build_dir, "cub-1.17.0.tar.gz"),
    #         )
    #         shutil.unpack_archive(zip_path, build_dir)
    #     extra_include_paths.append(target_path)
    #     extra_cuda_cflags.append("-DTHRUST_IGNORE_CUB_VERSION_CHECK")
    #     print(
    #         f"download cub because the cuda version is {cuda_toolkit_version()}"
    #     )

    if os.path.exists(os.path.join(build_dir, f"{name}.so")):
        # If the build exists, we assume the extension has been built
        # and we can load it.
        _C = load(
            name=name,
            sources=glob.glob(os.path.join(PATH, "csrc/*.cu")),
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_include_paths=extra_include_paths,
        )
    else:
        # Build from scratch. Remove the build directory just to be safe: pytorch jit might stuck
        # if the build directory exists.
        shutil.rmtree(build_dir)
        with Console().status(
            "[bold yellow]NerfAcc: Setting up CUDA (This may take a few minutes the first time)",
            spinner="bouncingBall",
        ):
            _C = load(
                name=name,
                sources=glob.glob(os.path.join(PATH, "csrc/*.cu")),
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=extra_include_paths,
            )
else:
    Console().print(
        "[yellow]NerfAcc: No CUDA toolkit found. NerfAcc will be disabled.[/yellow]"
    )


__all__ = ["_C"]

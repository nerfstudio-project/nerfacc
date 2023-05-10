import glob
import os
import os.path as osp
import platform
import sys

from setuptools import find_packages, setup

__version__ = None
exec(open("nerfacc/version.py", "r").read())

URL = "https://github.com/KAIR-BAIR/nerfacc"

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"


def get_ext():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(
        no_python_abi_suffix=True, use_ninja=False
    )


def get_extensions():
    import torch
    from torch.__config__ import parallel_info
    from torch.utils.cpp_extension import CUDAExtension

    extensions_dir = osp.join("nerfacc", "cuda", "csrc")
    sources = glob.glob(osp.join(extensions_dir, "*.cu")) + glob.glob(
        osp.join(extensions_dir, "*.cpp")
    )
    # remove generated 'hip' files, in case of rebuilds
    sources = [path for path in sources if "hip" not in path]

    undef_macros = []
    define_macros = []

    if sys.platform == "win32":
        define_macros += [("nerfacc_EXPORTS", None)]

    extra_compile_args = {"cxx": ["-O3"]}
    if not os.name == "nt":  # Not on Windows:
        extra_compile_args["cxx"] += ["-Wno-sign-compare"]
    extra_link_args = [] if WITH_SYMBOLS else ["-s"]

    info = parallel_info()
    if (
        "backend: OpenMP" in info
        and "OpenMP not found" not in info
        and sys.platform != "darwin"
    ):
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        if sys.platform == "win32":
            extra_compile_args["cxx"] += ["/openmp"]
        else:
            extra_compile_args["cxx"] += ["-fopenmp"]
    else:
        print("Compiling without OpenMP...")

    # Compile for mac arm64
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]

    nvcc_flags = os.getenv("NVCC_FLAGS", "")
    nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
    nvcc_flags += ["-O3"]
    if torch.version.hip:
        # USE_ROCM was added to later versions of PyTorch.
        # Define here to support older PyTorch versions as well:
        define_macros += [("USE_ROCM", None)]
        undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
    else:
        nvcc_flags += ["--expt-relaxed-constexpr"]
    extra_compile_args["nvcc"] = nvcc_flags

    extension = CUDAExtension(
        f"nerfacc.csrc",
        sources,
        include_dirs=[osp.join(extensions_dir, "include")],
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    return [extension]


# work-around hipify abs paths
include_package_data = True
# if torch.cuda.is_available() and torch.version.hip:
#     include_package_data = False

setup(
    name="nerfacc",
    version=__version__,
    description="A General NeRF Acceleration Toolbox",
    author="Ruilong",
    author_email="ruilongli94@gmail.com",
    url=URL,
    download_url=f"{URL}/archive/{__version__}.tar.gz",
    keywords=[],
    python_requires=">=3.7",
    install_requires=[
        "rich>=12",
        "torch",
        "typing_extensions; python_version<'3.8'",
    ],
    extras_require={
        # dev dependencies. Install them by `pip install nerfacc[dev]`
        "dev": [
            "black[jupyter]==22.3.0",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest==7.1.2",
            "pytest-xdist==2.5.0",
            "typeguard>=2.13.3",
            "pyyaml==6.0",
            "build",
            "twine",
            "ninja",
        ],
    },
    ext_modules=get_extensions() if not BUILD_NO_CUDA else [],
    cmdclass={"build_ext": get_ext()} if not BUILD_NO_CUDA else {},
    packages=find_packages(),
    include_package_data=include_package_data,
)

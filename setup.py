# # Environment flags to control different options
# #
# #   USE_MKL_BLAS=1
# #     enables use of MKL BLAS (requires PyTorch to be built with MKL support)

# import importlib
# import os
# import os.path as osp
# import subprocess
# import warnings

# from setuptools import Extension, find_packages, setup
# from setuptools.command.build_ext import build_ext

# __version__ = "0.1.0"
# URL = "https://github.com/KAIR-BAIR/nerfacc"


# class CMakeExtension(Extension):
#     def __init__(self, name, sourcedir=""):
#         Extension.__init__(self, name, sources=[])
#         self.sourcedir = os.path.abspath(sourcedir)


# class CMakeBuild(build_ext):
#     @staticmethod
#     def check_env_flag(name: str, default: str = "") -> bool:
#         value = os.getenv(name, default).upper()
#         return value in ["1", "ON", "YES", "TRUE", "Y"]

#     def get_ext_filename(self, ext_name):
#         # Remove Python ABI suffix:
#         ext_filename = super().get_ext_filename(ext_name)
#         ext_filename_parts = ext_filename.split(".")
#         ext_filename_parts = ext_filename_parts[:-2] + ext_filename_parts[-1:]
#         return ".".join(ext_filename_parts)

#     def build_extension(self, ext):
#         import sysconfig

#         import torch

#         extdir = os.path.abspath(osp.dirname(self.get_ext_fullpath(ext.name)))
#         self.build_type = "DEBUG" if self.debug else "RELEASE"
#         if self.debug is None:
#             if CMakeBuild.check_env_flag("DEBUG"):
#                 self.build_type = "DEBUG"
#             elif CMakeBuild.check_env_flag("REL_WITH_DEB_INFO"):
#                 self.build_type = "RELWITHDEBINFO"

#         if not osp.exists(self.build_temp):
#             os.makedirs(self.build_temp)

#         WITH_CUDA = torch.cuda.is_available()
#         WITH_CUDA = bool(int(os.getenv("FORCE_CUDA", WITH_CUDA)))

#         cmake_args = [
#             "-DUSE_PYTHON=ON",
#             f'-DWITH_CUDA={"ON" if WITH_CUDA else "OFF"}',
#             f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
#             f"-DCMAKE_BUILD_TYPE={self.build_type}",
#             f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
#         ]

#         if CMakeBuild.check_env_flag("USE_MKL_BLAS"):
#             include_dir = f"{sysconfig.get_path('data')}{os.sep}include"
#             cmake_args.append(f"-DBLAS_INCLUDE_DIR={include_dir}")
#             cmake_args.append("-DUSE_MKL_BLAS=ON")

#         with_ninja = importlib.util.find_spec("ninja") is not None
#         with_ninja |= os.environ.get("FORCE_NINJA") is not None
#         if with_ninja:
#             cmake_args += ["-GNinja"]
#         else:
#             warnings.warn(
#                 "Building times of 'pyg-lib' can be heavily improved"
#                 " by installing 'ninja': `pip install ninja`"
#             )

#         build_args = []

#         subprocess.check_call(
#             ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
#         )
#         subprocess.check_call(
#             ["cmake", "--build", "."] + build_args, cwd=self.build_temp
#         )


# install_requires = []

# ext_modules = [CMakeExtension("libpyg")]
# cmdclass = {"build_ext": CMakeBuild}

# setup(
#     name="nerfacc",
#     version=__version__,
#     description="A General NeRF Acceleration Toolbox",
#     author="Ruilong Li",
#     author_email="ruilongli94@gmail.com",
#     url=URL,
#     download_url=f"{URL}/archive/{__version__}.tar.gz",
#     keywords=["NeRF", "acceleration", "ray tracing", "rendering"],
#     python_requires=">=3.7",
#     install_requires=install_requires,
#     extras_require={},
#     packages=find_packages(),
#     ext_modules=ext_modules,
#     cmdclass=cmdclass,
# )


import glob
import os.path as osp

from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError:
    # Accomodate for sdist build
    from setuptools.command.build_ext import build_ext as BuildExtension
    from setuptools.extension import Extension as CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))

__version__ = None
exec(open("nerfacc/version.py", "r").read())

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []

ext_modules = [
    CUDAExtension(
        name="nerfacc.cuda",
        sources=glob.glob(
            osp.join(ROOT_DIR, "nerfacc", "cuda", "csrc", "*.cu")
        ),
        include_dirs=[osp.join(ROOT_DIR, "nerfacc", "cuda", "csrc", "include")],
    ),
]

setup(
    name="nerfacc",
    version=__version__,
    author="Ruilong Li",
    author_email="ruilongli94@gmail.com",
    description="A General NeRF Acceleration Toolbox",
    ext_modules=ext_modules,
    setup_requires=["pybind11>=2.5.0"],
    packages=["nerfacc", "nerfacc.cuda"],
    cmdclass={"build_ext": BuildExtension},
)

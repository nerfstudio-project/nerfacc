#!/bin/bash

# Took from https://github.com/pyg-team/pyg-lib/

case ${1} in
  cu118)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.8
    export PATH=${CUDA_HOME}/bin:$PATH
    export PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="6.0+PTX"
    ;;
  cu117)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.7
    export PATH=${CUDA_HOME}/bin:$PATH
    export PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="6.0+PTX"
    ;;
  cu116)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.6
    export PATH=${CUDA_HOME}/bin:$PATH
    export PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="6.0+PTX"
    ;;
  cu115)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.5
    export PATH=${CUDA_HOME}/bin:$PATH
    export PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="6.0+PTX"
    ;;
  cu113)
    export CUDA_HOME=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3
    export PATH=${CUDA_HOME}/bin:$PATH
    export PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
    export TORCH_CUDA_ARCH_LIST="6.0+PTX"
    ;;
  *)
    ;;
esac
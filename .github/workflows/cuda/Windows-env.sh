#!/bin/bash

# Took from https://github.com/pyg-team/pyg-lib/

case ${1} in
  cu118)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.8/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0+PTX"
    ;;
  cu117)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.7/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0+PTX"
    ;;
  cu116)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0+PTX"
    ;;
  cu115)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0+PTX"
    ;;
  cu113)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0+PTX"
    ;;
  *)
    ;;
esac
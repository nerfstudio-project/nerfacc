#!/bin/bash

case ${1} in
  cu117)
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.7/bin:${PATH}
    ;;
  # cu116 and cu115 have this issue: https://discuss.pytorch.org/t/cuda-11-6-extension-problem/145830
  # so we downgrade to cu113
  cu116)
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3/bin:${PATH}
    ;;
  cu115)
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3/bin:${PATH}
    ;;
  cu113)
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3/bin:${PATH}
    ;;
  *)
    ;;
esac
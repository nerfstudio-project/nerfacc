#!/bin/bash

case ${1} in
  cu117)
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.7/bin:${PATH}
    ;;
  cu116)
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.6/bin:${PATH}
    ;;
  cu115)
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.5/bin:${PATH}
    ;;
  cu113)
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3/bin:${PATH}
    ;;
  *)
    ;;
esac
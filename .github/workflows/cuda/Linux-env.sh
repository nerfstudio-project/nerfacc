#!/bin/bash

# Took from https://github.com/pyg-team/pyg-lib/

case ${1} in
  cu118)
    export PATH=/usr/local/cuda-11.8/bin:${PATH}
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu117)
    export PATH=/usr/local/cuda-11.7/bin:${PATH}
    export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu116)
    export PATH=/usr/local/cuda-11.6/bin:${PATH}
    export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:${LD_LIBRARY_PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu115)
    export PATH=/usr/local/cuda-11.5/bin:${PATH}
    export LD_LIBRARY_PATH=/usr/local/cuda-11.5/lib64:${LD_LIBRARY_PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu113)
    export PATH=/usr/local/cuda-11.3/bin:${PATH}
    export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:${LD_LIBRARY_PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu102)
    export PATH=/usr/local/cuda-10.2/bin:${PATH}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:${LD_LIBRARY_PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5"
    ;;
  *)
    ;;
esac
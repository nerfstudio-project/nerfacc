#!/bin/bash

# Took from https://github.com/pyg-team/pyg-lib/

case ${1} in
  cu118)
    CUDA_HOME=/usr/local/cuda-11.8
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu117)
    CUDA_HOME=/usr/local/cuda-11.7
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu116)
    CUDA_HOME=/usr/local/cuda-11.6
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu115)
    CUDA_HOME=/usr/local/cuda-11.5
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu113)
    CUDA_HOME=/usr/local/cuda-11.3
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu102)
    CUDA_HOME=/usr/local/cuda-10.2
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
    PATH=${CUDA_HOME}/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5"
    ;;
  *)
    ;;
esac
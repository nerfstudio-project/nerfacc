#!/bin/bash

case ${1} in
  cu117)
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    export PATH=/usr/local/cuda-11.7/bin:${PATH}
    ;;
  cu116)
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    export PATH=/usr/local/cuda-11.6/bin:${PATH}
    ;;
  cu115)
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    export PATH=/usr/local/cuda-11.5/bin:${PATH}
    ;;
  cu113)
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    export PATH=/usr/local/cuda-11.3/bin:${PATH}
    ;;
  cu102)
    export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5+PTX"
    export PATH=/usr/local/cuda-10.2/bin:${PATH}
    ;;
  *)
    ;;
esac
#!/bin/bash

# Took from https://github.com/pyg-team/pyg-lib/

case ${1} in
  cu118)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.8/bin:${PATH}
    ;;
  cu117)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.7/bin:${PATH}
    ;;
  cu116)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3/bin:${PATH}
    ;;
  cu115)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3/bin:${PATH}
    ;;
  cu113)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3/bin:${PATH}
    ;;
  *)
    ;;
esac
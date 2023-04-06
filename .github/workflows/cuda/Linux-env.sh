#!/bin/bash

# Took from https://github.com/pyg-team/pyg-lib/

case ${1} in
  cu118)
    export PATH=/usr/local/cuda-11.8/bin:${PATH}
    ;;
  cu117)
    export PATH=/usr/local/cuda-11.7/bin:${PATH}
    ;;
  cu116)
    export PATH=/usr/local/cuda-11.6/bin:${PATH}
    ;;
  cu115)
    export PATH=/usr/local/cuda-11.5/bin:${PATH}
    ;;
  cu113)
    export PATH=/usr/local/cuda-11.3/bin:${PATH}
    ;;
  cu102)
    export PATH=/usr/local/cuda-10.2/bin:${PATH}
    ;;
  *)
    ;;
esac
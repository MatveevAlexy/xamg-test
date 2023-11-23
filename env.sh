#!/bin/bash

function env_init_global {
  export PATH=/usr/bin:$PATH
  export LD_LIBRARY_PATH=/гr/lib/x86_64-linux-gnu/openmpi/lib:$LD_LIBRARY_PATH
  module load intel/2019
  module load cmake
  module load slurm
  export DNB_NOCUDA=1
  export MAKE_PARALLEL_LEVEL=4
}

#!/bin/bash
#SBATCH -J fbpic_control
#SBATCH -A gen0007
#SBATCH -C gpu
#SBATCH --qos regular
#SBATCH --time 01:00:00
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1

set -e

module load python cudatoolkit
source activate fbpic

export MPICH_GPU_SUPPORT_ENABLED=0
export FBPIC_ENABLE_GPUDIRECT=0

srun -n 8 python ionization_injection_runscript_00.py

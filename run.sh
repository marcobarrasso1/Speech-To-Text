#!/bin/bash

#SBATCH --job-name=RL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --partition=GPU
#SBATCH --output=out.txt
#SBATCH --error=err.err

source ../myenv/bin/activate

python3 -u train.py

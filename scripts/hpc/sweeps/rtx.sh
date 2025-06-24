#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p RTX
#SBATCH --time=72:00:00
#SBATCH -o logs/train/sweeps/%j/slrum_output.log
#SBATCH -e logs/train/sweeps/%j/slrum_error.log
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=8
#SBATCH --job-name=vda.1g

source .venv/bin/activate
wandb agent \<your-agent-id\>
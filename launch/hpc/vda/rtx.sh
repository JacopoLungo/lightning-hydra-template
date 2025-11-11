#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p RTX
#SBATCH --time=170:00:00
#SBATCH -o logs/train/sbatch/%j/slrum_output.log
#SBATCH -e logs/train/sbatch/%j/slrum_error.log
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=8
#SBATCH --job-name=vda.rtx

source .venv/bin/activate
python src/train.py \
    experiment=vda \
    trainer.max_epochs=100 \
    run_name="$SLURM_JOB_ID" \
    type_run_name=sbatch \

# sbatch out path should be: logs/${task_name}/sbatch/%j/...
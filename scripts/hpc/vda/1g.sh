#!/bin/bash
#SBATCH --gpus=1g.20gb:1
#SBATCH -p A100
#SBATCH --time=72:00:00
#SBATCH -o logs/train/sbatch/%j/slrum_output.log
#SBATCH -e logs/train/sbatch/%j/slrum_error.log
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=8
#SBATCH --job-name=cls.1g

source .venv/bin/activate
python src/train.py \
    experiment=pureforest \
    trainer.max_epochs=100 \
    run_name=$SLURM_JOB_ID \
    type_run_name=sbatch

# sbatch out path should be: logs/${task_name}/sbatch/%j/...
#!/bin/bash
#SBATCH --partition=long  #unkillable #main #long
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --mem=64G
#SBATCH --time=4:00:00    
#SBATCH --job-name=test-exps

module load python/3.10
source ~/CrediNet/.venv/bin/activate

uv run python experiments.py


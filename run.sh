#!/bin/bash
#SBATCH --partition=long  #unkillable #main #long
#SBATCH --output=logs/dec-%j.out
#SBATCH --error=logs/dec-%j.err
#SBATCH --cpus-per-task=4                    # Ask for 4 CPUs
#SBATCH --mem=256G
#SBATCH --time=168:00:00    
#SBATCH --job-name=dec           # The job will run for 1 day

# if [ $# -lt 1 ]; then 
#     echo "Usage: $0 <start-month> [<end-month>]"
#     echo "e.g.: $0 'January 2025' 'March 2025'."
#     exit 1
# elif [ $# -eq 1 ]; then
#     START_MONTH="$1"
#     END_MONTH="$1"
# elif [ $# -eq 2 ]; then 
#     START_MONTH="$1"
#     END_MONTH="$2"
# fi

# export PATH="$HOME/bin:$PATH"
module load python/3.10
source ~/CrediNet/.venv/bin/activate

python scripts/compute_graph_stats.py \
  --month-url https://huggingface.co/datasets/credi-net/CrediBench/tree/main/dec2024 \
  --month 2024-12 \
  --update months.json

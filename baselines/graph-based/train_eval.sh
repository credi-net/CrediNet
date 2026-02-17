#!/bin/bash
#SBATCH --partition=long
#SBATCH --output=logs/GNN-c-%j.out
#SBATCH --error=logs/GNN-c-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=GNN-c

set -euo pipefail

###############################################################################
# TRAINING & EVALUATION - GNN:
#   sbatch exps/run.sh gnn-train-eval --task regression --output-dir ~/scratch/models/gnn_regression
#   sbatch exps/run.sh gnn-train-eval --task classification --output-dir ~/scratch/models/gnn_classification
#
# TRAINING & EVALUATION - LGBM:
#   sbatch exps/run.sh lgbm-train-eval --task regression --output-dir ~/scratch/models/lgbm_regression
#   sbatch exps/run.sh lgbm-train-eval --task classification --output-dir ~/scratch/models/lgbm_classification
###############################################################################

cd ~/CrediNet/exps

module load cuda/11.8
module load python/3.10

source ~/CrediNet/.venv/bin/activate

export HF_HOME=~/scratch/hf_cache
export HF_HUB_CACHE=~/scratch/hub
export MODELS_HOME=~/scratch/models

if [ "${1:-}" = "gnn-train-eval" ]; then
    shift  
    echo "[START] GNN Train, Eval @ $(date)"
    python train_eval_gnn.py "$@"
    
elif [ "${1:-}" = "lgbm-train-eval" ]; then
    shift  
    echo "[START] LGBM Train, Eval @ $(date)"
    python train_eval_lgbm.py "$@"
else
    echo "[ERR] Unknown ${1:-}"
fi

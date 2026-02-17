#!/bin/bash
#SBATCH --partition=long  
#SBATCH --output=logs/LGBM-test-%j.out
#SBATCH --error=logs/LGBM-test-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=168:00:00    
#SBATCH --job-name=test-LGBM

set -euo pipefail
cd ~/CrediNet/exps

module load python/3.10
source ~/CrediNet/.venv/bin/activate

export HF_HOME=~/scratch/hf_cache
export HF_HUB_CACHE=~/scratch/hub
export MODELS_HOME=~/scratch/models

echo "==================================================================="
echo "LightGBM Training and Evaluation Pipeline"
echo "==================================================================="

CLASSIFICATION_MONTH="${CLASSIFICATION_MONTH:-oct2024}"

LGBM_OUTPUT="$MODELS_HOME/lgbm_baseline_${CLASSIFICATION_MONTH}"
FEATURES="data/domain_features.csv"
LABELS="data/regression/processed/domain_pc1.csv"

if [ ! -f "$FEATURES" ]; then
    echo "[ERR] Features file not found: $FEATURES"
    echo "You need to extract features first (e.g., PageRank, degree, etc.)"
    exit 1
fi

if [ ! -f "$LABELS" ]; then
    echo "[ERR] Labels file not found: $LABELS"
    exit 1
fi

echo "[START - Train LightGBM] $(date)."
echo "  Output: $LGBM_OUTPUT"

mkdir -p "$LGBM_OUTPUT"

python exps/train_lgbm.py \
    --features "$FEATURES" \
    --labels "$LABELS" \
    --output "$LGBM_OUTPUT" \
    --classification-month "$CLASSIFICATION_MONTH" \
    --classification-splits-dir "data/splits"

echo "[END - Train LightGBM] $(date)."

echo "[START - Eval. LightGBM] $(date)."

LGBM_MODEL="$LGBM_OUTPUT/lgbm_model.txt"

if [ ! -f "$LGBM_MODEL" ]; then
    echo "[ERR] LightGBM model not found: $LGBM_MODEL"
    echo "Training may have failed."
    exit 1
fi

echo "[LGBM] Submitting reg. and class. evals."
JOB=$(sbatch --parsable exps/run.sh lgbm \
    --output-dir "eval_results/lgbm")

echo "[LGBM] Evaluation job: $JOB"  
echo ""
echo "==================================================================="
echo "Pipeline Summary"
echo "==================================================================="
echo "LGBM Model:    $LGBM_MODEL"
echo "Evaluation:    Job $JOB"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results in:   eval_results/lgbm/"
echo "==================================================================="

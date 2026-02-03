#!/bin/bash
#SBATCH --partition=long  #unkillable #main #long
#SBATCH --output=logs/GNN-test-%j.out
#SBATCH --error=logs/GNN-test-%j.err
#SBATCH --cpus-per-task=4                     # Ask for 4 CPUs
#SBATCH --mem=64G
#SBATCH --time=168:00:00    
#SBATCH --job-name=test-GNN

set -euo pipefail
cd ~/CrediNet/exps

module load cuda/11.8
module load python/3.10
source ~/CrediNet/.venv/bin/activate

export HF_HOME=~/scratch/hf_cache
export HF_HUB_CACHE=~/scratch/hub
export MODELS_HOME=~/scratch/models

echo "==================================================================="
echo "GNN Training and Eval Pipeline"
echo "==================================================================="

CLASSIFICATION_MONTH="${CLASSIFICATION_MONTH:-oct2024}"

GNN_OUTPUT="$MODELS_HOME/gnn_baseline_${CLASSIFICATION_MONTH}"
ATTRIBUTES="data/domain_attributes.csv"
LINKS="data/domain_links.csv"
LABELS="data/regression/processed/domain_pc1.csv"

if [ ! -f "$ATTRIBUTES" ]; then
    echo "[ERR] Attributes file not found: $ATTRIBUTES"
    echo "Need to prepare graph data first."
    exit 1
fi

if [ ! -f "$LINKS" ]; then
    echo "[ERR] Links file not found: $LINKS"
    echo "Need to prepare graph data first."
    exit 1
fi

if [ ! -f "$LABELS" ]; then
    echo "[ERR] Labels file not found: $LABELS"
    exit 1
fi

echo "[START - Train GNN] $(date)"
echo "  Output: $GNN_OUTPUT"

python exps/train_gnn.py \
    --attributes "$ATTRIBUTES" \
    --links "$LINKS" \
    --labels "$LABELS" \
    --output "$GNN_OUTPUT" \
    --classification-month "$CLASSIFICATION_MONTH" \
    --classification-splits-dir "data/splits"

echo "[END - Train GNN] $(date)"

echo "[START - Eval. GNN] $(date)"
GNN_PREDICTIONS="$GNN_OUTPUT/test_predictions_regression.csv"
if [ ! -f "$GNN_PREDICTIONS" ]; then
    echo "Error: GNN predictions not found: $GNN_PREDICTIONS"
    echo "Training may have failed."
    exit 1
fi

echo "[GNN] Submitting reg. and class. evals."
JOB=$(sbatch --parsable run.sh gnn \
    --gnn-predictions "$GNN_PREDICTIONS" \
    --output-dir "eval_results/gnn")

echo "[GNN] Evaluation job: $JOB"

echo ""
echo "==================================================================="
echo "Pipeline Summary"
echo "==================================================================="
echo "GNN Model:     $GNN_OUTPUT"
echo "Predictions:   $GNN_PREDICTIONS"
echo "Evaluation:    Job $JOB"
echo ""
echo "Results in:   eval_results/gnn/"
echo "==================================================================="

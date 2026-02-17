#!/bin/bash
#SBATCH --partition=long
#SBATCH --output=logs/LGBM-%j.out
#SBATCH --error=logs/LGBM-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=LGBM

set -euo pipefail

###############################################################################
# UNIFIED CREDI-NET EVALUATION LAUNCHER
#
# TRAINING (run these first):
#   python exps/train_gnn.py --attributes data/attrs.csv --links data/links.csv --labels data/labels.csv --output models/gnn
#   python exps/train_lgbm.py --features data/features.csv --labels data/labels.csv --output models/lgbm
#
# EVALUATION - MY_MODEL (uses scores parquet):
#   sbatch exps/run.sh my_model --scores-parquet path/to/scores.parquet
#     # Runs both regression (MAE) and classification (Acc) on per-month test sets
#   sbatch exps/run.sh my_model --scores-parquet path/to/scores.parquet --task regression
#   sbatch exps/run.sh my_model --scores-parquet path/to/scores.parquet --task classification
#
# EVALUATION - LLM:
#   sbatch exps/run.sh llm --model gpt-4o-mini              # Both tasks, all months
#   sbatch exps/run.sh llm --model gpt-4o-mini --web-search # With web search
#   sbatch exps/run.sh llm --model gemini-2.0-flash         # Gemini model
#   sbatch exps/run.sh llm --model gpt-4o-mini --months "oct2024,dec2024"  # Specific months
#
# EVALUATION - LLM-REGRESSION (Test set only with MAE, Max(AE), Refusal Rate):
#   sbatch exps/run.sh llm-regression --provider openai --model gpt-5-mini --output-dir exps/eval_results/llm-gpt-5-mini-regression
#   sbatch exps/run.sh llm-regression --provider openai --model gpt-5-mini --output-dir exps/eval_results/llm-gpt-5-mini-regression --web-search
#   sbatch exps/run.sh llm-regression --provider openai --model gpt-4o-mini --output-dir exps/eval_results/llm-gpt-4o-mini-regression
#   # Outputs: individual .txt files + evaluation_summary.json + evaluation_results.txt
#
# TRAINING & EVALUATION - GNN:
#   sbatch exps/run.sh gnn-train-eval --task regression --output-dir ~/scratch/models/gnn_regression
#   sbatch exps/run.sh gnn-train-eval --task classification --output-dir ~/scratch/models/gnn_classification
#
# TRAINING & EVALUATION - LGBM:
#   sbatch exps/run.sh lgbm-train-eval --task regression --output-dir ~/scratch/models/lgbm_regression
#   sbatch exps/run.sh lgbm-train-eval --task classification --output-dir ~/scratch/models/lgbm_classification
#
# LEGACY EVALUATION - GNN & LGBM (requires pre-trained models):
#   sbatch exps/run.sh gnn --gnn-predictions models/gnn/test_predictions.csv
#   sbatch exps/run.sh lgbm --lgbm-model models/lgbm/lgbm_model.txt --test-features data/test_features.csv
#
# REFUSAL HANDLING:
#   Regression (MAE): LLM refusals → predicted value 0.5 (neutral)
#   Classification (Acc): LLM refusals → predicted reliability 0.5 (neutral)
#   Refusal counts reported in metrics; does NOT reduce coverage
#
###############################################################################

# Change to the exps directory
# When run via sbatch, we need to explicitly cd to where the script lives
cd ~/CrediNet/exps

# Load modules
module load cuda/11.8
module load python/3.10

# Activate environment (adjust path as needed)
if [ -f ~/CrediNet/.venv/bin/activate ]; then
    source ~/CrediNet/.venv/bin/activate
elif [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found"
    exit 1
fi

# Set environment variables for caching
export HF_HOME=~/scratch/hf_cache
export HF_HUB_CACHE=~/scratch/hub
export MODELS_HOME=~/scratch/models

# Handle different task types
if [ "${1:-}" = "gnn-train-eval" ]; then
    shift  # Remove 'gnn-train-eval' from arguments
    echo "Running GNN training and evaluation..."
    python train_eval_gnn.py "$@"
    
elif [ "${1:-}" = "lgbm-train-eval" ]; then
    shift  # Remove 'lgbm-train-eval' from arguments
    echo "Running LGBM training and evaluation..."
    python train_eval_lgbm.py "$@"
    
elif [ "${1:-}" = "llm-regression" ]; then
    shift  # Remove 'llm-regression' from arguments
    echo "Running LLM evaluation on regression test set..."
    python evaluate_llm_regression.py "$@"
    
else
    # Run the unified evaluation pipeline for other tasks
    python eval_pipeline.py "$@"
fi

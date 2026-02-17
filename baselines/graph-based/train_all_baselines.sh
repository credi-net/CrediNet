#!/bin/bash

set -e

echo "[SUBMIT] @ $(date)"

# GNN
for month in oct2024 nov2024 dec2024; do
  CLASSIFICATION_MONTH=$month sbatch train_and_eval_gnn.sh
  echo "[SUBMITTED] GNN for $month"
  sleep 2
done

# LGBM
for month in oct2024 nov2024 dec2024; do
  echo "[SUBMITTED] LGBM for $month"
  CLASSIFICATION_MONTH=$month sbatch train_and_eval_lgbm.sh
  sleep 2
done

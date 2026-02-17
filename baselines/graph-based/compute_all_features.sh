#!/bin/bash
#SBATCH --job-name=compute_features
#SBATCH --output=logs/compute_features_%j.out
#SBATCH --error=logs/compute_features_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

set -e

MONTH="oct2024"

echo "[START - Get domains for feature-based training] $(date)"
python get_domains.py 
echo "[END - Get domains for feature-based training] $(date)"


echo "[START - Graph Features] $(date)"

python compute_graph_features.py \
  $MONTH \
  data/all_domains_for_features.csv \
  data/

echo "[END - Graph Features] $(date)"

ls -lh data/domain_features.csv data/domain_links.csv data/domain_attributes.csv

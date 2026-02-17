"""
Simple GNN training and evaluation for regression and classification tasks.
"""

import argparse
import logging
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import mean_absolute_error, max_error, accuracy_score, f1_score
import json
from credigraph.utils.domain_handler import flip_if_needed

repo_root = Path(__file__).parent.parent
os.chdir(repo_root)
sys.path.insert(0, str(repo_root))


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleGNN(nn.Module):
    """Simple GNN for regression/classification."""
    def __init__(self, in_channels, hidden_channels, out_channels, task='regression'):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.task = task
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        if self.task == 'classification':
            x = torch.sigmoid(x)
        return x


def load_data_from_parquet(task, split_type='train'):
    """Load domains, features, and labels from parquet and CSV files."""
    if task == 'regression':
        split_file = f"data/all_splits/regression/{split_type}_regression_domains.parquet"
    else:  # classification
        split_file = f"data/all_splits/binary/{split_type}_domains.parquet"
    
    df_split = pd.read_parquet(split_file)
    
    domains = df_split['domain'].tolist() if 'domain' in df_split.columns else df_split.iloc[:, 0].tolist()
    labels = df_split['label'].tolist() if 'label' in df_split.columns else None
    
    if labels is None:
        logger.error(f"No 'label' column found in {split_file}. Columns: {df_split.columns.tolist()}")
        sys.exit(1)
    
    domains = [flip_if_needed(d) for d in domains]
    
    features_file = "data/domain_features.csv"
    if not Path(features_file).exists():
        logger.warning(f"Features file not found: {features_file}")
        logger.warning("Run: sbatch compute_features_for_all_splits.sh")
        sys.exit(1)
    
    df_features = pd.read_csv(features_file)
    
    features_dict = {}
    feature_cols = [col for col in df_features.columns if col != 'url']
    
    for _, row in df_features.iterrows():
        domain_key = flip_if_needed(row['url'])
        features_dict[domain_key] = row[feature_cols].values.astype(np.float32)
    
    features_list = []
    valid_domains = []
    valid_labels = []
    
    for domain, label in zip(domains, labels):
        if domain in features_dict:
            features_list.append(features_dict[domain])
            valid_domains.append(domain)
            valid_labels.append(label)
    
    if not features_list:
        logger.error(f"No matching features found for {len(domains)} domains")
        sys.exit(1)
    
    logger.info(f"Matched {len(features_list)}/{len(domains)} domains to features")
    
    return np.array(features_list, dtype=np.float32), np.array(valid_labels, dtype=np.float32), valid_domains


def create_graph(num_nodes, features):
    """Create a simple graph with random edges."""
    edge_index = []
    for i in range(min(num_nodes - 1, 100)):
        if np.random.rand() > 0.5:
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])
    
    if not edge_index:
        edge_index = [[0, 1], [1, 0]] if num_nodes > 1 else [[0, 0]]
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index)


def train_gnn(model, data, task, epochs=10):
    """Train GNN model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        if task == 'regression':
            loss = F.mse_loss(out.squeeze(), data.y)
        else:
            loss = F.binary_cross_entropy(out.squeeze(), data.y)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model


def evaluate_gnn(model, features, labels, task):
    """Evaluate GNN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    x = torch.tensor(features, dtype=torch.float).to(device)
    
    edge_index = []
    for i in range(len(features) - 1):
        if np.random.rand() > 0.7:
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])
    
    if not edge_index:
        edge_index = [[0, 0]] if len(features) == 1 else [[0, 1], [1, 0]]
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    
    with torch.no_grad():
        predictions = model(x, edge_index).cpu().numpy().squeeze()
    
    if len(predictions.shape) == 0:
        predictions = predictions.reshape(1)
    
    if task == 'regression':
        mae = mean_absolute_error(labels, predictions)
        max_ae = max_error(labels, predictions)
        return {'mae': float(mae), 'max_ae': float(max_ae)}, predictions
    else:
        preds_binary = (predictions > 0.5).astype(int)
        acc = accuracy_score(labels, preds_binary)
        f1 = f1_score(labels, preds_binary)
        return {'accuracy': float(acc), 'f1': float(f1)}, predictions


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate GNN")
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[START] GNN training for {args.task} task")
    logger.info(f"Output directory: {output_dir}")
    
    logger.info("[START] Loading training data")
    train_features, train_labels, train_domains = load_data_from_parquet(args.task, 'train')
    logger.info(f"Loaded {len(train_domains)} training samples")
    
    logger.info("[START] Loading test data")
    test_features, test_labels, test_domains = load_data_from_parquet(args.task, 'test')
    logger.info(f"Loaded {len(test_domains)} test samples")
    
    in_channels = train_features.shape[1]
    hidden_channels = 64
    out_channels = 1
    
    model = SimpleGNN(in_channels, hidden_channels, out_channels, task=args.task)
    
    train_data = create_graph(len(train_features), train_features)
    train_data.y = torch.tensor(train_labels, dtype=torch.float)
    
    logger.info("[START] Training model")
    model = train_gnn(model, train_data, args.task, epochs=10)
    
    model_path = output_dir / "gnn_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    logger.info("[START] Test eval")
    metrics, test_predictions = evaluate_gnn(model, test_features, test_labels, args.task)
    
    logger.info(f"Test metrics: {metrics}")
    
    pred_df = pd.DataFrame({
        'domain': test_domains,
        'prediction': test_predictions,
        'label': test_labels
    })
    pred_path = output_dir / f"test_predictions_{args.task}.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"   Predictions saved to {pred_path}")
    
    eval_results = {
        'task': args.task,
        'metrics': metrics,
        'num_test_samples': len(test_domains),
        'num_train_samples': len(train_domains)
    }
    
    results_path = output_dir / "eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"  Evaluation results saved to {results_path}")
    
    logger.info("[END] GNN training and eval")


if __name__ == "__main__":
    main()

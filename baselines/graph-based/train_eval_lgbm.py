"""
Simple LGBM training and evaluation for regression and classification tasks.
"""

import argparse
import logging
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, max_error, accuracy_score, f1_score
import json
from credigraph.utils.domain_handler import flip_if_needed

repo_root = Path(__file__).parent.parent
os.chdir(repo_root)
sys.path.insert(0, str(repo_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def train_lgbm(X_train, y_train, task, num_leaves=31):
    """Train LGBM model."""
    params = {
        'num_leaves': num_leaves,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'device_type': 'cpu',  # Force CPU
        'num_threads': 4,
        'verbose': -1,
    }
    
    if task == 'regression':
        model = lgb.LGBMRegressor(**params)
    else:
        model = lgb.LGBMClassifier(**params)
    
    model.fit(X_train, y_train)
    return model


def evaluate_lgbm(model, X_test, y_test, task):
    """Evaluate LGBM model."""
    predictions = model.predict(X_test)
    
    if task == 'regression':
        mae = mean_absolute_error(y_test, predictions)
        max_ae = max_error(y_test, predictions)
        return {'mae': float(mae), 'max_ae': float(max_ae)}, predictions
    else:
        if hasattr(model, 'predict_proba'):
            preds_proba = model.predict_proba(X_test)[:, 1]
            preds_binary = (preds_proba > 0.5).astype(int)
        else:
            preds_binary = predictions
        acc = accuracy_score(y_test, preds_binary)
        f1 = f1_score(y_test, preds_binary)
        return {'accuracy': float(acc), 'f1': float(f1)}, preds_binary


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate LGBM")
    parser.add_argument("--task", choices=["regression", "classification"], required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    print(f"[START - LGBM {args.task}] @ {datetime.now().isoformat()}")
    
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[START] LGBM training for {args.task} task")
    logger.info(f"Output directory: {output_dir}")
    
    print(f"[START] Loading training data for {args.task} @ {datetime.now().isoformat()}", flush=True)
    logger.info("[START] Loading training data")
    X_train, y_train, train_domains = load_data_from_parquet(args.task, 'train')
    print(f" Loaded {len(train_domains)} training samples", flush=True)
    logger.info(f"Loaded {len(train_domains)} training samples")
    
    print(f"[START] Loading test data for {args.task} @ {datetime.now().isoformat()}", flush=True)
    logger.info("[START] Loading test data")
    X_test, y_test, test_domains = load_data_from_parquet(args.task, 'test')
    print(f"  Loaded {len(test_domains)} test samples", flush=True)
    logger.info(f"Loaded {len(test_domains)} test samples")
    
    logger.info("[START] Training LGBM model")
    model = train_lgbm(X_train, y_train, args.task)
    
    model_path = output_dir / "lgbm_model.txt"
    model.booster_.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    logger.info(f"[START] Test eval @ {datetime.now().isoformat()}")
    metrics, test_predictions = evaluate_lgbm(model, X_test, y_test, args.task)
    
    logger.info(f"Test metrics: {metrics}")
    
    pred_df = pd.DataFrame({
        'domain': test_domains,
        'prediction': test_predictions,
        'label': y_test
    })
    pred_path = output_dir / f"test_predictions_{args.task}.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved to {pred_path}")
    
    eval_results = {
        'task': args.task,
        'metrics': metrics,
        'num_test_samples': len(test_domains),
        'num_train_samples': len(train_domains)
    }
    
    results_path = output_dir / "eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Evaluation results saved to {results_path}")
    
    logger.info(f"[END] LGBM training and eval @ {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
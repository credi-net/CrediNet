import os
import pandas as pd
import json
from typing import Tuple, List, Dict
from pydantic import BaseModel, Field
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path

from utils.test_sets import resolve_test_set_path


def load_test_domains_with_pc1(test_set_path: str = "creditext_test_with_pc1.csv") -> Tuple[List[str], Dict[str, float]]:
    """
    Load test domains with pc1 scores from parquet (domains) and regression CSV (pc1).
    Handles reversed domain formats and normalizes for matching.
    
    Args:
        test_set_path: Path to the parquet file containing test domain names
        
    Returns:
        Tuple of (domains_list, ground_truth_labels_dict)
        where domains_list is a list of domain names
        and ground_truth_labels_dict maps domain -> pc1 score
    """
    resolved_path = resolve_test_set_path(test_set_path)

    print(f"[INFO] Loading test domains from {resolved_path}...")
    df_domains = pd.read_parquet(resolved_path)
    
    # Handle different column name formats
    domain_col = None
    if 'domain' in df_domains.columns:
        domain_col = 'domain'
    elif 'Domains_test' in df_domains.columns:
        domain_col = 'Domains_test'
    else:
        raise ValueError(f"Parquet file missing domain column. Found: {df_domains.columns.tolist()}")
    
    # Normalize domains (flip reversed format if needed)
    from credigraph.utils.domain_handler import flip_if_needed, normalize_domain
    domains_list = [flip_if_needed(d) for d in df_domains[domain_col].tolist()]
    
    # Load pc1 scores from regression CSV
    pc1_path = "../data/regression/processed/domain_pc1.csv"
    if not Path(pc1_path).exists():
        pc1_path = "data/regression/processed/domain_pc1.csv"
    
    print(f"[INFO] Loading pc1 scores from {pc1_path}...")
    df_pc1 = pd.read_csv(pc1_path)
    
    if 'domain' not in df_pc1.columns or 'pc1' not in df_pc1.columns:
        raise ValueError(f"PC1 CSV must have 'domain' and 'pc1' columns. Found: {df_pc1.columns.tolist()}")
    
    # Normalize pc1 domains as well (flip if needed)
    df_pc1['domain_normalized'] = df_pc1['domain'].apply(flip_if_needed)
    
    # Create ground truth dict from pc1 scores (using normalized domains)
    ground_truth_labels = dict(zip(df_pc1['domain_normalized'], df_pc1['pc1']))
    
    # Filter to only domains in test set (both already normalized)
    ground_truth_labels = {d: ground_truth_labels.get(d) for d in domains_list if d in ground_truth_labels}
    
    print(f"[INFO] Loaded {len(domains_list)} test domains, {len(ground_truth_labels)} with pc1 scores")
    
    return domains_list, ground_truth_labels


def get_domains_from_credibench(output_root):
    """Load domains from CrediBench dataset."""
    dataset = load_dataset(
        "credi-net/CrediBench",
        data_files="jan2025/vertices.csv.gz",
        split="train"
    )

    # Extract domains
    domains = set(row["domain"] for row in dataset if row.get("domain"))

    # Optional: skip already-processed domains
    existing = {
        os.path.splitext(f)[0]
        for f in os.listdir(output_root)
        if f.endswith(".txt")
    }

    return sorted(domains - existing)


def get_domains_from_creditext(output_root):
    """
    Load domains from CrediText DQR active domains dataset.
    Source: https://huggingface.co/datasets/credi-net/CrediText
    File: cc-content/dqr/dqr_active_domains.csv
    """
    print("[INFO] Loading domains from CrediText DQR dataset...")
    
    dataset = load_dataset(
        "credi-net/CrediText",
        data_files="cc-content/dqr/dqr_active_domains.csv",
        split="train"
    )
    
    # Extract domains (assuming the CSV has a 'domain' column)
    # If the column name is different, adjust accordingly
    domains = set(row["domain"] for row in dataset if row.get("domain"))
    
    print(f"[INFO] Loaded {len(domains)} domains from CrediText DQR")
    
    # Skip already-processed domains
    existing = {
        os.path.splitext(f)[0]
        for f in os.listdir(output_root)
        if f.endswith(".txt")
    }
    
    remaining = sorted(domains - existing)
    print(f"[INFO] {len(existing)} already processed, {len(remaining)} remaining")
    
    return remaining


def train_valid_test_split(
    target: str, labeled_11k_df: pd.DataFrame, test_valid_size: float = 0.4
) -> Tuple[
    pd.DataFrame, List[float], pd.DataFrame, List[float], pd.DataFrame, List[float]
]:
    """
    Perform stratified train-validation-test split on labeled domains.
    
    Args:
        target: Target variable name ('pc1' or 'mbfc_bias')
        labeled_11k_df: DataFrame with 'domain' column and target column
        test_valid_size: Proportion of data for test+validation (default 0.4)
        
    Returns:
        Tuple of (X_train, y_train, X_valid, y_valid, X_test, y_test)
        where X_* are DataFrames with 'domain' and target columns
        and y_* are lists of target values
    """
    if target == 'pc1':
        quantiles = labeled_11k_df[target].quantile([0.2, 0.4, 0.6, 0.8, 1.0])
    elif target == 'mbfc_bias':
        quantiles = labeled_11k_df[target].quantile([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    else:
        raise ValueError(f"Unsupported target: {target}. Must be 'pc1' or 'mbfc_bias'")
    
    bins = [labeled_11k_df[target].min()] + quantiles.tolist()
    labeled_11k_df[target + '_cat'] = pd.cut(
        labeled_11k_df[target], bins=bins, labels=quantiles, include_lowest=True
    )
    
    X = labeled_11k_df[['domain', target]]
    y = labeled_11k_df[target + '_cat']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_valid_size, stratify=y, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test, y_test, test_size=0.5, stratify=y_test, random_state=42
    )
    
    return (
        X_train,
        labeled_11k_df.iloc[X_train.index][target].tolist(),
        X_valid,
        labeled_11k_df.iloc[X_valid.index][target].tolist(),
        X_test,
        labeled_11k_df.iloc[X_test.index][target].tolist(),
    )


class DomainRating(BaseModel):
    rating: float = Field(
        description="Domain credibility rating, ranging from -1 to 1",
        ge=-1,
        le=1,
    )
    url: str = Field(description="The URL.")
    explanation: str = Field(description="Explanation of the rating.")


def load_api_key(api_key_path, provider):
    with open(api_key_path) as f:
        key_obj = json.load(f)
        if provider.lower() not in key_obj:
            raise ValueError(
                f"Unknown provider: {provider}, must be one of {key_obj.keys()}"
            )

        api_key = key_obj.get(provider)
        return api_key


def get_domain_to_query(url_list_path, output_root):
    """Load domains from a local CSV or parquet file."""
    path = Path(url_list_path)
    if path.suffix.lower() == ".parquet":
        domain_df = pd.read_parquet(path)
    else:
        domain_df = pd.read_csv(path, usecols=["domain"])

    if "domain" not in domain_df.columns:
        raise ValueError(f"Domain list must include a 'domain' column. Found: {domain_df.columns.tolist()}")

    from credigraph.utils.domain_handler import flip_if_needed
    domain_df["domain"] = domain_df["domain"].astype(str).apply(flip_if_needed)
    processed_domains = set()
    for file_name in os.listdir(output_root):
        if file_name.endswith(".txt"):
            processed_domains.add(file_name[:-4])

    print(f"{len(processed_domains)} domains have been processed")

    domain_to_query = domain_df[~domain_df.domain.isin(processed_domains)].domain.tolist()
    return domain_to_query


def get_test_domains_from_creditext(output_root, target='pc1', test_valid_size=0.4, return_labels=False, test_set_path: str = "exps/creditext_test_with_pc1.csv"):
    """
    Load test domains from pre-created CSV with pc1 scores.
    
    Args:
        output_root: Directory containing already-processed domains
        target: Target variable ('pc1' or 'mbfc_bias')
        test_valid_size: Proportion of data for test+validation (default 0.4)
        return_labels: If True, return tuple of (domains, labels_dict)
        
    Returns:
        List of domain names in the test set (excluding already processed)
        OR tuple of (domains, labels_dict) if return_labels=True
    """
    print(f"[INFO] Loading test domains from pre-created CSV...")
    print(f"[INFO] Target variable: {target}")
    
    # Load the pre-created test set with pc1 scores
    resolved_path = resolve_test_set_path(test_set_path)
    df = pd.read_csv(resolved_path)
    
    # Check if required columns exist
    if 'domain' not in df.columns:
        raise ValueError("Dataset must have 'domain' column")
    if target not in df.columns:
        raise ValueError(f"Dataset must have '{target}' column")
    
    # Remove rows with missing target values
    df = df.dropna(subset=[target])
    
    print(f"[INFO] Loaded {len(df)} test domains with {target} labels")
    
    # Get already processed domains
    processed_domains = set()
    if os.path.exists(output_root):
        processed_domains = set(os.listdir(output_root))
    
    # Filter out already processed domains
    df = df[~df['domain'].isin(processed_domains)]
    print(f"[INFO] After excluding {len(processed_domains)} already processed: {len(df)} domains remain")
    
    # Get test domains
    test_domains = set(df['domain'].tolist())
    
    # Create ground truth labels dict if needed
    ground_truth_labels = None
    if return_labels:
        ground_truth_labels = dict(zip(df['domain'].tolist(), df[target].tolist()))
    
    print(f"[INFO] Test set has {len(test_domains)} domains")
    
    remaining = sorted(test_domains)
    print(f"[INFO] {len(remaining)} domains ready for LLM querying")
    
    if return_labels:
        return remaining, ground_truth_labels
    return remaining


def evaluate_predictions(output_root: str, ground_truth_labels: dict, target: str, threshold: float = 0.5):
    """
    Evaluate LLM predictions against ground truth labels.
    Computes TWO sets of metrics:
      1. Only on answered predictions (excludes -1 refusals)
      2. With worst-case penalty for refusals (assigns 0 or 1 to maximize error)
    
    Args:
        output_root: Directory containing prediction files
        ground_truth_labels: Dict mapping domain -> ground truth score
        target: Target variable name ('pc1' or 'mbfc_bias')
        threshold: Threshold for binary classification (default 0.5)
        
    Returns:
        Tuple of (accuracy, metrics_answered_only, metrics_with_penalty)
    """
    import json
    from json_repair import repair_json
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error, 
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix
    )
    
    # For answered-only metrics
    predictions_answered = []
    ground_truth_answered = []
    
    # For penalty metrics (all predictions)
    predictions_penalty = []
    ground_truth_penalty = []
    
    domains_evaluated = []
    parse_errors = 0
    refusals_count = 0
    
    # Parse all prediction files
    for domain, true_score in ground_truth_labels.items():
        pred_file = os.path.join(output_root, f"{domain}.txt")
        
        if not os.path.exists(pred_file):
            continue
            
        try:
            with open(pred_file, 'r') as f:
                content = f.read()
                
            # Try to parse JSON (repair if needed)
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                content = repair_json(content)
                result = json.loads(content)
            
            # Extract rating
            rating = result.get('rating', -1)
            
            # Track for answered-only metrics
            if rating != -1:
                predictions_answered.append(rating)
                ground_truth_answered.append(true_score)
            else:
                refusals_count += 1
            
            # For penalty metrics: assign neutral prediction (0.5) if refused
            if rating == -1:
                penalty_rating = 0.5  # Neutral/uncertain prediction
            else:
                penalty_rating = rating
            
            predictions_penalty.append(penalty_rating)
            ground_truth_penalty.append(true_score)
            domains_evaluated.append(domain)
            
        except Exception as e:
            parse_errors += 1
            continue
    
    print(f"[INFO] Successfully parsed {len(predictions_penalty)} predictions")
    print(f"[INFO] Answered: {len(predictions_answered)}, Refused: {refusals_count}, Parse errors: {parse_errors}")
    
    if len(predictions_penalty) == 0:
        print("[WARNING] No valid predictions found!")
        return 0.0, {}, {}
    
    # Compute metrics for ANSWERED ONLY
    metrics_answered = {}
    if len(predictions_answered) > 0:
        preds_ans = pd.Series(predictions_answered)
        gt_ans = pd.Series(ground_truth_answered)
        
        mae_ans = mean_absolute_error(gt_ans, preds_ans)
        mse_ans = mean_squared_error(gt_ans, preds_ans)
        max_ae_ans = (preds_ans - gt_ans).abs().max()
        corr_ans = preds_ans.corr(gt_ans)
        
        thresh_ans = gt_ans.median()
        y_true_ans = (gt_ans >= thresh_ans).astype(int)
        y_pred_ans = (preds_ans >= thresh_ans).astype(int)
        
        acc_ans = accuracy_score(y_true_ans, y_pred_ans)
        prec_ans, rec_ans, f1_ans, _ = precision_recall_fscore_support(
            y_true_ans, y_pred_ans, average='binary', zero_division=0
        )
        # Handle case where only one class is predicted
        cm_ans = confusion_matrix(y_true_ans, y_pred_ans, labels=[0, 1])
        tn_ans, fp_ans, fn_ans, tp_ans = cm_ans.ravel()
        
        metrics_answered = {
            'Accuracy': acc_ans,
            'Precision': prec_ans,
            'Recall': rec_ans,
            'F1-Score': f1_ans,
            'MSE': mse_ans,
            'MAE': mae_ans,
            'Max(AE)': max_ae_ans,
            'Correlation': corr_ans,
            'Coverage': len(predictions_answered) / len(ground_truth_labels),
            'Sample_Count': len(predictions_answered),
            'Refusals': refusals_count,
            'True_Positives': tp_ans,
            'True_Negatives': tn_ans,
            'False_Positives': fp_ans,
            'False_Negatives': fn_ans,
        }
    
    # Compute metrics WITH PENALTY for refusals
    preds_pen = pd.Series(predictions_penalty)
    gt_pen = pd.Series(ground_truth_penalty)
    
    mae_pen = mean_absolute_error(gt_pen, preds_pen)
    mse_pen = mean_squared_error(gt_pen, preds_pen)
    max_ae_pen = (preds_pen - gt_pen).abs().max()
    corr_pen = preds_pen.corr(gt_pen)
    
    thresh_pen = gt_pen.median()
    y_true_pen = (gt_pen >= thresh_pen).astype(int)
    y_pred_pen = (preds_pen >= thresh_pen).astype(int)
    
    acc_pen = accuracy_score(y_true_pen, y_pred_pen)
    prec_pen, rec_pen, f1_pen, _ = precision_recall_fscore_support(
        y_true_pen, y_pred_pen, average='binary', zero_division=0
    )
    # Handle case where only one class is predicted
    cm_pen = confusion_matrix(y_true_pen, y_pred_pen, labels=[0, 1])
    tn_pen, fp_pen, fn_pen, tp_pen = cm_pen.ravel()
    
    metrics_penalty = {
        'Accuracy': acc_pen,
        'Precision': prec_pen,
        'Recall': rec_pen,
        'F1-Score': f1_pen,
        'MSE': mse_pen,
        'MAE': mae_pen,
        'Max(AE)': max_ae_pen,
        'Correlation': corr_pen,
        'Coverage': len(predictions_penalty) / len(ground_truth_labels),
        'Sample_Count': len(predictions_penalty),
        'Refusals': refusals_count,
        'True_Positives': tp_pen,
        'True_Negatives': tn_pen,
        'False_Positives': fp_pen,
        'False_Negatives': fn_pen,
    }
    
    # Return both sets of metrics
    return acc_pen, metrics_answered, metrics_penalty
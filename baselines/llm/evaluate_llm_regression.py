"""
Query LLM for test set domains and evaluate results in real-time.
Supports both regression and classification modes.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score

import api_factory
import prompt_factory
from credigraph.utils.domain_handler import flip_if_needed


def load_regression_test_set():
    """Load regression test set with ground truth labels."""
    base_dir = Path(__file__).parent.parent.parent
    test_path = base_dir / 'data' / 'all_splits' / 'regression' / 'test_regression_domains.parquet'
    
    print(f"Loading regression test set from {test_path}")
    df = pd.read_parquet(test_path)
    
    ground_truth = {}
    for _, row in df.iterrows():
        domain = flip_if_needed(row['domain'])
        ground_truth[domain] = row['label']
    
    print(f"Loaded {len(ground_truth)} domains with ground truth labels")
    return ground_truth


def load_classification_test_set():
    """Load classification balanced test set with ground truth labels."""
    base_dir = Path(__file__).parent.parent.parent
    test_path = base_dir / 'data' / 'all_splits' / 'binary' / 'balanced' / 'test_domains.parquet'
    
    print(f"Loading classification balanced test set from {test_path}")
    df = pd.read_parquet(test_path)
    
    ground_truth = {}
    for _, row in df.iterrows():
        domain = flip_if_needed(row['domain'])
        ground_truth[domain] = row['label']
    
    print(f"Loaded {len(ground_truth)} domains with ground truth labels")
    return ground_truth


def parse_llm_response(response_text):
    """
    Parse LLM response to extract rating.
    Returns (rating, is_refusal) tuple.
    """
    try:
        data = json.loads(response_text)
        rating = data.get('rating')
        if rating is not None:
            rating_float = float(rating)
            if rating_float == -1:
                return 0.5, True  
            clamped = max(0.0, min(1.0, rating_float))
            return clamped, False
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return 0.5, True 


def evaluate_predictions(predictions, ground_truth, mode='regression', threshold=0.5):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Dict mapping domain -> (rating, is_refusal)
        ground_truth: Dict mapping domain -> true_label
        mode: 'regression' or 'classification'
        threshold: Threshold for binary classification (default 0.5)
        
    Returns:
        Dict with evaluation metrics
    """
    num_matched = 0
    num_refused = 0
    num_total = len(predictions)
    
    y_true = []
    y_pred = []
    
    for domain, (pred_rating, is_refusal) in predictions.items():  
        if domain in ground_truth:
            true_rating = ground_truth[domain]
            if pred_rating == -1:
                num_refused += 1
                pred_rating = 0.5  
            
            y_true.append(true_rating)
            y_pred.append(pred_rating)
            num_matched += 1
    
    refusal_rate = num_refused / num_total if num_total > 0 else 0
    
    metrics = {
        'num_total': num_total,
        'num_matched': num_matched,
        'num_refused': num_refused,
        'refusal_rate': refusal_rate,
        'coverage': num_matched / len(ground_truth) if ground_truth else 0
    }
    
    if not y_true:
        metrics.update({'mae': None, 'max_ae': None, 'accuracy': None, 'f1': None})
        return metrics
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if mode == 'classification':
        # Binary classification metrics
        y_true_binary = (y_true >= threshold).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        
        metrics.update({
            'accuracy': accuracy,
            'f1': f1,
            'mae': None,
            'max_ae': None
        })
    else:
        # Regression metrics
        absolute_errors = np.abs(y_true - y_pred)
        mae = np.mean(absolute_errors)
        max_ae = np.max(absolute_errors)
        
        metrics.update({
            'mae': mae,
            'max_ae': max_ae,
            'accuracy': None,
            'f1': None
        })
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM on test set domains (regression or classification)."
    )
    parser.add_argument("--provider", required=True, help="API provider ('openai' or 'vllm')")
    parser.add_argument("--model", required=True, help="Model name to query")
    parser.add_argument("--output-dir", required=True, help="Directory for output files")
    parser.add_argument(
        "--web-search",
        action="store_true",
        default=False,
        help="Enable web search for OpenAI API"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['regression', 'classification'],
        default='regression',
        help="Evaluation mode: regression or classification"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary classification (default 0.5)"
    )
    
    args = parser.parse_args()
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") if args.provider == "openai" else None
    
    api_client = api_factory.create_api_client(
        args.provider,
        api_key,
        model_name=args.model,
        enable_web_search=args.web_search
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load appropriate test set based on mode
    if args.mode == 'classification':
        ground_truth = load_classification_test_set()
    else:
        ground_truth = load_regression_test_set()
    
    domains_to_query = sorted(ground_truth.keys())
    
    existing = {
        f.stem for f in output_dir.glob("*.txt")
        if f.name not in ['evaluation_results.txt', 'evaluation_summary.json']
    }
    
    new_domains = [d for d in domains_to_query if d not in existing]
    
    print(f"\nTotal regression domains: {len(domains_to_query)}")
    print(f"Already evaluated: {len(existing)}")
    print(f"Remaining to query: {len(new_domains)}")
    if args.web_search and args.provider == "openai":
        print("Web search is ENABLED\n")
    
    predictions = {}
    
    print(f"[START - Load existing predictions from {len(existing)} already-processed domains] @ {datetime.now().isoformat()}")
    for domain in existing:
        txt_file = output_dir / f"{domain}.txt"
        try:
            with open(txt_file, 'r') as f:
                response = f.read()
                rating, is_refusal = parse_llm_response(response)
                predictions[domain] = (rating, is_refusal)
        except Exception:
            predictions[domain] = (0.5, True)
    
    print(f"[START - LLM Query for {len(new_domains)} domains] @ {datetime.now().isoformat()}")
    
    for domain in tqdm(new_domains, desc="Querying LLM"):
        system_prompt = prompt_factory.SYS_BASE
        user_instruction = f"{prompt_factory.USER_INSTRUCTION.format(domain=domain)} {prompt_factory.USER_FORMAT}"
        
        try:
            result = api_client.query_model(args.model, system_prompt, user_instruction)
            
            # Save result
            with open(output_dir / f"{domain}.txt", "w") as f:
                f.write(result)
            
            # Parse and store prediction
            rating, is_refusal = parse_llm_response(result)
            predictions[domain] = (rating, is_refusal)
            
        except Exception as e:
            print(f"\nError querying {domain}: {e}")
            predictions[domain] = (0.5, True)
    
    print(f"[START] Evaluating predictions @ {datetime.now().isoformat()}")
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS ({args.mode.upper()})")
    print("="*80)
    
    metrics = evaluate_predictions(predictions, ground_truth, mode=args.mode, threshold=args.threshold)
    
    print(f"Total domains: {metrics['num_total']}")
    print(f"Matched with ground truth: {metrics['num_matched']}")
    print(f"Refused/Invalid: {metrics['num_refused']}")
    print(f"Refusal rate: {metrics['refusal_rate']:.2%}")
    print(f"Coverage: {metrics['coverage']:.2%}")
    
    if args.mode == 'classification':
        if metrics['accuracy'] is not None:
            print(f"\nAccuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
        else:
            print("\nNo valid predictions to evaluate")
    else:
        if metrics['mae'] is not None:
            print(f"\nMAE: {metrics['mae']:.4f}")
            print(f"Max(AE): {metrics['max_ae']:.4f}")
        else:
            print("\nNo valid predictions to evaluate")
    
    summary = {
        'model': args.model,
        'provider': args.provider,
        'web_search': args.web_search,
        'mode': args.mode,
        'threshold': args.threshold if args.mode == 'classification' else None,
        'metrics': metrics
    }
    
    summary_path = output_dir / 'evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    
    results_path = output_dir / 'evaluation_results.txt'
    with open(results_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Provider: {args.provider}\n")
        f.write(f"Web Search: {args.web_search}\n")
        f.write(f"Mode: {args.mode}\n")
        if args.mode == 'classification':
            f.write(f"Threshold: {args.threshold}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total domains: {metrics['num_total']}\n")
        f.write(f"Matched with ground truth: {metrics['num_matched']}\n")
        f.write(f"Refused/Invalid: {metrics['num_refused']}\n")
        f.write(f"Refusal rate: {metrics['refusal_rate']:.2%}\n")
        f.write(f"Coverage: {metrics['coverage']:.2%}\n")
        if args.mode == 'classification':
            if metrics['accuracy'] is not None:
                f.write(f"\nAccuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        else:
            if metrics['mae'] is not None:
                f.write(f"\nMAE: {metrics['mae']:.4f}\n")
                f.write(f"Max(AE): {metrics['max_ae']:.4f}\n")
    
    print(f"[END] Results saved to {results_path}")


if __name__ == "__main__":
    main()

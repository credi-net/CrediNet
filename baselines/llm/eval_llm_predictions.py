"""
LLM Eval (Reg).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from credigraph.utils.domain_handler import flip_if_needed


def load_ground_truth(parquet_path: str) -> Dict[str, float]:
    """
    Load ground truth labels from parquet file.
    Returns dict mapping normal domain format -> label.
    """
    df = pd.read_parquet(parquet_path)
    
    ground_truth = {}
    for _, row in df.iterrows():
        domain = flip_if_needed(row['domain'])
        ground_truth[domain] = row['label']
    
    return ground_truth


def load_llm_predictions(predictions_dir: Path) -> Dict[str, float]:
    """
    Load all LLM predictions from a directory.
    
    Args:
        predictions_dir: Path to directory containing .txt prediction files
        
    Returns:
        Dictionary mapping domain -> rating (refusals set to -1)
    """
    predictions = {}
    
    if not predictions_dir.exists():
        return predictions
    
    for txt_file in predictions_dir.glob('*.txt'):
        domain = txt_file.stem
        
        try:
            with open(txt_file, 'r') as f:
                data = json.load(f)
                rating = data.get('rating')
                
                if rating is not None:
                    rating_float = float(rating)
                    if rating_float == -1:
                        predictions[domain] = -1
                    else:
                        predictions[domain] = max(0.0, min(1.0, rating_float))
                else:
                    predictions[domain] = -1
                    
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            predictions[domain] = -1
    
    return predictions


def evaluate_predictions(predictions, ground_truth):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Dict mapping domain -> (rating)
        ground_truth: Dict mapping domain -> true_label
        
    Returns:
        Dict with evaluation metrics
    """
    absolute_errors = []
    num_matched = 0
    num_refused = 0
    num_total = len(predictions)
    
    for domain, pred_rating in predictions.items():
        if domain in ground_truth:
            true_rating = ground_truth[domain]
            if pred_rating == -1:
                num_refused += 1
                pred_rating = 0.5  
            ae = abs(pred_rating - true_rating)
            absolute_errors.append(ae)
            num_matched += 1
    
    if absolute_errors:
        mae = np.mean(absolute_errors)
        max_ae = np.max(absolute_errors)
    else:
        mae = None
        max_ae = None
    
    refusal_rate = num_refused / num_total if num_total > 0 else 0
    
    return {
        'mae': mae,
        'max_ae': max_ae,
        'num_total': num_total,
        'num_matched': num_matched,
        'num_refused': num_refused,
        'refusal_rate': refusal_rate,
        'coverage': num_matched / len(ground_truth) if ground_truth else 0
    }
 

def find_llm_directories(eval_results_dir: Path, output_dir: Path, month: str = 'oct2024') -> List[Tuple[str, Path]]:
    """
    Find all LLM prediction directories for a given month.
    
    Args:
        eval_results_dir: Path to eval_results directory
        output_dir: Path to output_dir directory
        month: Month to evaluate (e.g., 'oct2024')
        
    Returns:
        List of tuples (llm_name, predictions_path)
    """
    llm_dirs = []
    
    for base_dir in [eval_results_dir, output_dir]:
        if not base_dir.exists():
            continue
            
        for llm_dir in base_dir.iterdir():
            if not llm_dir.is_dir():
                continue
            
            if base_dir == output_dir:
                if llm_dir.name.startswith('gpt') or llm_dir.name.startswith('gemini'):
                    llm_name = f"llm-{llm_dir.name}"
                    llm_dirs.append((llm_name, llm_dir))
                continue
            
            if not llm_dir.name.startswith('llm-'):
                continue
            
            month_dir = llm_dir / month
            if not month_dir.exists():
                continue
            
            for subdir in month_dir.iterdir():
                if subdir.is_dir():
                    llm_dirs.append((llm_dir.name, subdir))
                    break
    
    return llm_dirs


def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    eval_results_dir = base_dir / 'exps' / 'eval_results'
    output_dir = base_dir / 'exps' / 'output_dir'
    nested_eval_results = base_dir / 'exps' / 'exps' / 'eval_results'
    
    test_sets = {
        'regression': base_dir / 'data' / 'all_splits' / 'regression' / 'test_regression_domains.parquet',
        'binary': base_dir / 'data' / 'all_splits' / 'binary' / 'test_domains.parquet',
        'binary_balanced': base_dir / 'data' / 'all_splits' / 'binary' / 'balanced' / 'test_domains.parquet',
    }
    
    for test_set_name, ground_truth_path in test_sets.items():
        print("\n" + "="*80)
        print(f"EVALUATING AGAINST: {test_set_name.upper()}")
        print("="*80)
        
        print(f"Loading ground truth from {ground_truth_path}")
        ground_truth = load_ground_truth(ground_truth_path)
        print(f"Ground truth: {len(ground_truth)} domains\n")
        
        month = 'oct2024'
        print(f"Finding LLM predictions for {month}...\n")
        llm_dirs = find_llm_directories(eval_results_dir, output_dir, month)
        
        if nested_eval_results.exists():
            for llm_dir in nested_eval_results.iterdir():
                if llm_dir.is_dir() and llm_dir.name.startswith('llm-'):
                    llm_dirs.append((llm_dir.name, llm_dir))
        
        if not llm_dirs:
            print(f"No LLM directories found for {month}")
            continue
        
        print(f"Found {len(llm_dirs)} LLM models:\n")
        
        results = []
        
        for llm_name, predictions_dir in sorted(llm_dirs):
            print(f"Evaluating {llm_name}...")
            print(f"  Predictions dir: {predictions_dir}")
            
            predictions = load_llm_predictions(predictions_dir)
            
            metrics = evaluate_predictions(predictions, ground_truth)
            
            results.append({
                'llm_name': llm_name,
                'mae': metrics['mae'],
                'max_ae': metrics['max_ae'],
                'num_predictions': metrics['num_total'],
                'num_matched': metrics['num_matched'],
                'refusals': metrics['num_refused'],
                'refusal_rate': metrics['refusal_rate'],
                'coverage': metrics['coverage']
            })
            
            if metrics['mae'] is not None:
                print(f"  MAE: {metrics['mae']:.4f}")
                print(f"  Max(AE): {metrics['max_ae']:.4f}")
                print(f"  Refusals: {metrics['num_refused']}/{metrics['num_total']} ({metrics['refusal_rate']*100:.2f}%)")
                print(f"  Coverage: {metrics['num_matched']}/{len(ground_truth)} ({metrics['coverage']*100:.1f}%)")
            else:
                print(f"  No matching predictions found!")
            print()
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('mae')
        
        print("\n" + "="*80)
        print(f"SUMMARY - {month} - {test_set_name.upper()}")
        print("="*80)
        print(results_df.to_string(index=False))
        print()
        
        output_path = base_dir / 'exps' / f'llm_eval_{month}_{test_set_name}.csv'
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}\n")


if __name__ == '__main__':
    main()

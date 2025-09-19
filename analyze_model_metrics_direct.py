#!/usr/bin/env python3
"""
Script to analyze model metrics from the Comorbidities-Detection models directory.
Extracts training progress information and generates a comprehensive CSV summary.
"""

import os
import re
import csv
import json
import glob
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import statistics

def extract_auroc_from_line(line: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Extract Train Avg AUROC, Val Avg AUROC, Train Median AUROC, and Val Median AUROC from a log line.
    
    Args:
        line: Log line containing AUROC information
        
    Returns:
        Tuple of (train_avg_auroc, val_avg_auroc, train_median_auroc, val_median_auroc) or (None, None, None, None) if not found
    """
    # Pattern to match: "Train Avg AUROC: 0.8079, Val Avg AUROC: 0.8116"
    avg_pattern = r'Train Avg AUROC: ([\d.]+), Val Avg AUROC: ([\d.]+)'
    avg_match = re.search(avg_pattern, line)
    
    # Pattern to match: "Train Median AUROC: 0.7771, Val Median AUROC: 0.4614"
    median_pattern = r'Train Median AUROC: ([\d.]+), Val Median AUROC: ([\d.]+)'
    median_match = re.search(median_pattern, line)
    
    train_avg_auroc = float(avg_match.group(1)) if avg_match else None
    val_avg_auroc = float(avg_match.group(2)) if avg_match else None
    train_median_auroc = float(median_match.group(1)) if median_match else None
    val_median_auroc = float(median_match.group(2)) if median_match else None
    
    return train_avg_auroc, val_avg_auroc, train_median_auroc, val_median_auroc

def extract_biomarker_metrics(log_lines: List[str], best_epoch_line_idx: int) -> List[Dict[str, Any]]:
    """
    Extract validation metrics per biomarker from the log lines.
    
    Args:
        log_lines: List of log file lines
        best_epoch_line_idx: Index of the line with the best validation AUROC
        
    Returns:
        List of dictionaries containing biomarker metrics
    """
    biomarker_metrics = []
    
    # Look for the "Validation metrics per biomarker:" line after the best epoch
    for i in range(best_epoch_line_idx + 1, min(best_epoch_line_idx + 10, len(log_lines))):
        line = log_lines[i]
        
        if "Validation metrics per biomarker:" in line:
            # Parse the next few lines for biomarker metrics
            for j in range(i + 1, min(i + 15, len(log_lines))):
                metric_line = log_lines[j].strip()
                
                # Skip empty lines
                if not metric_line:
                    continue
                    
                # Check if this is a new epoch or other non-biomarker line (stops biomarker parsing)
                if ("Starting Epoch" in metric_line or 
                    ("Epoch" in metric_line and "completed" in metric_line) or
                    "🎯" in metric_line or
                    "No improvement" in metric_line or
                    "Early stopping" in metric_line):
                    break
                
                # Check if this line contains biomarker metrics (contains colon and AUROC)
                if ":" in metric_line and "AUROC=" in metric_line:
                    # Parse biomarker metric line
                    # Format: "2025-09-15 19:44:45,857 -   CALCIUMSCORING_ABDOMINALAGATSTON_BINARY: AUROC=0.8172, Acc=0.7248, F1=0.5449"
                    # or "  BIOMARKER_NAME: MSE=654.4478, MAE=21.0701" for regression
                    try:
                        # Remove timestamp prefix if present
                        if " - " in metric_line:
                            clean_line = metric_line.split(" - ", 1)[1].strip()
                        else:
                            clean_line = metric_line.strip()
                            
                        if ":" not in clean_line:
                            continue
                            
                        biomarker_name, metrics_str = clean_line.split(":", 1)
                        biomarker_name = biomarker_name.strip()
                        
                        # Parse metrics
                        metrics = {}
                        if "AUROC=" in metrics_str:
                            auroc_match = re.search(r'AUROC=([\d.]+)', metrics_str)
                            if auroc_match:
                                metrics['AUROC'] = float(auroc_match.group(1))
                        
                        if "Acc=" in metrics_str:
                            acc_match = re.search(r'Acc=([\d.]+)', metrics_str)
                            if acc_match:
                                metrics['Accuracy'] = float(acc_match.group(1))
                        
                        if "F1=" in metrics_str:
                            f1_match = re.search(r'F1=([\d.]+)', metrics_str)
                            if f1_match:
                                metrics['F1'] = float(f1_match.group(1))
                        
                        if "MSE=" in metrics_str:
                            mse_match = re.search(r'MSE=([\d.]+)', metrics_str)
                            if mse_match:
                                metrics['MSE'] = float(mse_match.group(1))
                        
                        if "MAE=" in metrics_str:
                            mae_match = re.search(r'MAE=([\d.]+)', metrics_str)
                            if mae_match:
                                metrics['MAE'] = float(mae_match.group(1))
                        
                        if metrics:  # Only add if we found at least one metric
                            biomarker_metrics.append({
                                'biomarker': biomarker_name,
                                **metrics
                            })
                            
                    except Exception as e:
                        print(f"Warning: Could not parse biomarker metric line: {metric_line}")
                        continue
            
            break
    
    return biomarker_metrics

def find_best_epoch_metrics(log_file_path: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], List[Dict[str, Any]]]:
    """
    Find the best validation median AUROC and extract corresponding metrics.
    
    Args:
        log_file_path: Path to the training progress log file
        
    Returns:
        Tuple of (best_val_avg_auroc, best_train_avg_auroc, best_val_median_auroc, best_train_median_auroc, biomarker_metrics)
    """
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading log file {log_file_path}: {e}")
        return None, None, None, None, []
    
    best_val_avg_auroc = 0.0
    best_train_avg_auroc = 0.0
    best_val_median_auroc = 0.0
    best_train_median_auroc = 0.0
    best_epoch_line_idx = -1
    
    # Find the best validation median AUROC (using median as the selection criterion)
    # Look for both average and median AUROC in consecutive lines
    for i in range(len(lines) - 1):
        # Check current line for average AUROC
        train_avg, val_avg, _, _ = extract_auroc_from_line(lines[i])
        # Check next line for median AUROC
        _, _, train_median, val_median = extract_auroc_from_line(lines[i + 1])
        
        if val_median is not None and val_median > best_val_median_auroc:
            best_val_median_auroc = val_median
            best_train_median_auroc = train_median if train_median is not None else 0.0
            best_val_avg_auroc = val_avg if val_avg is not None else 0.0
            best_train_avg_auroc = train_avg if train_avg is not None else 0.0
            best_epoch_line_idx = i + 1  # Use the median line as reference
    
    # Extract biomarker metrics for the best epoch
    biomarker_metrics = []
    if best_epoch_line_idx >= 0:
        biomarker_metrics = extract_biomarker_metrics(lines, best_epoch_line_idx)
    
    return best_val_avg_auroc, best_train_avg_auroc, best_val_median_auroc, best_train_median_auroc, biomarker_metrics

def calculate_median_auroc(biomarker_metrics: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[str]]:
    """
    Calculate the median AUROC from biomarker metrics.
    
    Args:
        biomarker_metrics: List of biomarker metric dictionaries
        
    Returns:
        Tuple of (median_auroc, biomarker_name_with_median_auroc)
    """
    auroc_values = []
    auroc_biomarkers = []
    
    for metric in biomarker_metrics:
        if 'AUROC' in metric:
            auroc_values.append(metric['AUROC'])
            auroc_biomarkers.append(metric['biomarker'])
    
    if not auroc_values:
        return None, None
    
    median_auroc = statistics.median(auroc_values)
    
    # Find the biomarker closest to the median
    median_biomarker = None
    min_diff = float('inf')
    
    for i, auroc in enumerate(auroc_values):
        diff = abs(auroc - median_auroc)
        if diff < min_diff:
            min_diff = diff
            median_biomarker = auroc_biomarkers[i]
    
    return median_auroc, median_biomarker

def scan_models_directory(models_dir: str) -> List[Dict[str, Any]]:
    """
    Scan the models directory and extract metrics from all model runs.
    
    Args:
        models_dir: Path to the models directory
        
    Returns:
        List of dictionaries containing model metrics
    """
    results = []
    
    # Check if the given directory contains model directories directly
    direct_model_dirs = [d for d in os.listdir(models_dir) 
                        if os.path.isdir(os.path.join(models_dir, d)) 
                        and not d.startswith('.')
                        and '_lr' in d]  # Model directories typically contain '_lr'
    
    if direct_model_dirs:
        # Process models directly in the given directory
        print(f"Processing models directly in directory: {models_dir}")
        for model_dir in direct_model_dirs:
            model_path = os.path.join(models_dir, model_dir)
            log_file = os.path.join(model_path, 'logs', 'training_progress.log')
            
            if os.path.exists(log_file):
                print(f"  Processing model: {model_dir}")
                val_avg_auroc, train_avg_auroc, val_median_auroc, train_median_auroc, biomarker_metrics = find_best_epoch_metrics(log_file)
                
                if val_median_auroc is not None:
                    median_auroc, median_biomarker = calculate_median_auroc(biomarker_metrics)
                    
                    result = {
                        'biomarker_config_type': 'direct',  # No specific config type for direct models
                        'model_name': model_dir,
                        'best_val_avg_auroc': val_avg_auroc,
                        'best_train_avg_auroc': train_avg_auroc,
                        'best_val_median_auroc': val_median_auroc,
                        'best_train_median_auroc': train_median_auroc,
                        'biomarker_median_auroc': median_auroc,
                        'biomarker_median_auroc_biomarker': median_biomarker,
                        'biomarker_metrics': biomarker_metrics,
                        'flags': []  # No flags for direct models
                    }
                    results.append(result)
        return results
    
    # If no direct model directories, look for biomarker config type directories
    biomarker_config_dirs = [d for d in os.listdir(models_dir) 
                           if os.path.isdir(os.path.join(models_dir, d)) 
                           and not d.startswith('.')]
    
    for config_type in biomarker_config_dirs:
        config_path = os.path.join(models_dir, config_type)
        print(f"Processing biomarker config: {config_type}")
        
        # Check for subdirectories (flags)
        subdirs = [d for d in os.listdir(config_path) 
                  if os.path.isdir(os.path.join(config_path, d)) 
                  and not d.startswith('.')]
        
        # Check if there are model directories directly in the config type
        model_dirs = [d for d in os.listdir(config_path) 
                     if os.path.isdir(os.path.join(config_path, d)) 
                     and not d.startswith('.')
                     and '_lr' in d]  # Model directories typically contain '_lr'
        
        if model_dirs:
            # Process models directly in config type directory
            for model_dir in model_dirs:
                model_path = os.path.join(config_path, model_dir)
                log_file = os.path.join(model_path, 'logs', 'training_progress.log')
                
                if os.path.exists(log_file):
                    print(f"  Processing model: {model_dir}")
                    val_avg_auroc, train_avg_auroc, val_median_auroc, train_median_auroc, biomarker_metrics = find_best_epoch_metrics(log_file)
                    
                    if val_median_auroc is not None:
                        median_auroc, median_biomarker = calculate_median_auroc(biomarker_metrics)
                        
                        result = {
                            'biomarker_config_type': config_type,
                            'model_name': model_dir,
                            'best_val_avg_auroc': val_avg_auroc,
                            'best_train_avg_auroc': train_avg_auroc,
                            'best_val_median_auroc': val_median_auroc,
                            'best_train_median_auroc': train_median_auroc,
                            'biomarker_median_auroc': median_auroc,
                            'biomarker_median_auroc_biomarker': median_biomarker,
                            'biomarker_metrics': biomarker_metrics,
                            'flags': []  # No flags for direct models
                        }
                        results.append(result)
        
        # Process subdirectories (flags)
        for subdir in subdirs:
            subdir_path = os.path.join(config_path, subdir)
            
            # Check if this subdir contains model directories
            subdir_model_dirs = [d for d in os.listdir(subdir_path) 
                               if os.path.isdir(os.path.join(subdir_path, d)) 
                               and not d.startswith('.')
                               and '_lr' in d]
            
            if subdir_model_dirs:
                print(f"  Processing subdirectory: {subdir}")
                
                # Check for nested subdirectories (like linear_probe/regularized)
                nested_subdirs = [d for d in os.listdir(subdir_path) 
                                if os.path.isdir(os.path.join(subdir_path, d)) 
                                and not d.startswith('.')
                                and '_lr' not in d]
                
                if nested_subdirs:
                    # Process nested subdirectories
                    for nested_subdir in nested_subdirs:
                        nested_path = os.path.join(subdir_path, nested_subdir)
                        nested_model_dirs = [d for d in os.listdir(nested_path) 
                                           if os.path.isdir(os.path.join(nested_path, d)) 
                                           and not d.startswith('.')
                                           and '_lr' in d]
                        
                        for model_dir in nested_model_dirs:
                            model_path = os.path.join(nested_path, model_dir)
                            log_file = os.path.join(model_path, 'logs', 'training_progress.log')
                            
                            if os.path.exists(log_file):
                                print(f"    Processing nested model: {model_dir}")
                                val_avg_auroc, train_avg_auroc, val_median_auroc, train_median_auroc, biomarker_metrics = find_best_epoch_metrics(log_file)
                                
                                if val_median_auroc is not None:
                                    median_auroc, median_biomarker = calculate_median_auroc(biomarker_metrics)
                                    
                                    result = {
                                        'biomarker_config_type': config_type,
                                        'model_name': model_dir,
                                        'best_val_avg_auroc': val_avg_auroc,
                                        'best_train_avg_auroc': train_avg_auroc,
                                        'best_val_median_auroc': val_median_auroc,
                                        'best_train_median_auroc': train_median_auroc,
                                        'biomarker_median_auroc': median_auroc,
                                        'biomarker_median_auroc_biomarker': median_biomarker,
                                        'biomarker_metrics': biomarker_metrics,
                                        'flags': [subdir, nested_subdir]
                                    }
                                    results.append(result)
                else:
                    # Process models directly in subdirectory
                    for model_dir in subdir_model_dirs:
                        model_path = os.path.join(subdir_path, model_dir)
                        log_file = os.path.join(model_path, 'logs', 'training_progress.log')
                        
                        if os.path.exists(log_file):
                            print(f"    Processing model: {model_dir}")
                            val_avg_auroc, train_avg_auroc, val_median_auroc, train_median_auroc, biomarker_metrics = find_best_epoch_metrics(log_file)
                            
                            if val_median_auroc is not None:
                                median_auroc, median_biomarker = calculate_median_auroc(biomarker_metrics)
                                
                                result = {
                                    'biomarker_config_type': config_type,
                                    'model_name': model_dir,
                                    'best_val_avg_auroc': val_avg_auroc,
                                    'best_train_avg_auroc': train_avg_auroc,
                                    'best_val_median_auroc': val_median_auroc,
                                    'best_train_median_auroc': train_median_auroc,
                                    'biomarker_median_auroc': median_auroc,
                                    'biomarker_median_auroc_biomarker': median_biomarker,
                                    'biomarker_metrics': biomarker_metrics,
                                    'flags': [subdir]
                                }
                                results.append(result)
    
    return results

def write_results_to_csv(results: List[Dict[str, Any]], output_path: str):
    """
    Write results to CSV file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output CSV file
    """
    if not results:
        print("No results to write.")
        return
    
    # Get all possible flag columns
    all_flags = set()
    for result in results:
        all_flags.update(result['flags'])
    
    # Define CSV columns
    columns = [
        'biomarker_config_type',
        'model_name',
        'best_val_avg_auroc',
        'best_train_avg_auroc',
        'best_val_median_auroc',
        'best_train_median_auroc',
        'biomarker_median_auroc',
        'biomarker_median_auroc_biomarker',
        'biomarker_metrics_json',
        'num_biomarkers'
    ]
    
    # Add flag columns
    for flag in sorted(all_flags):
        columns.append(f'flag_{flag}')
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        
        for result in results:
            row = {
                'biomarker_config_type': result['biomarker_config_type'],
                'model_name': result['model_name'],
                'best_val_avg_auroc': result['best_val_avg_auroc'],
                'best_train_avg_auroc': result['best_train_avg_auroc'],
                'best_val_median_auroc': result['best_val_median_auroc'],
                'best_train_median_auroc': result['best_train_median_auroc'],
                'biomarker_median_auroc': result['biomarker_median_auroc'],
                'biomarker_median_auroc_biomarker': result['biomarker_median_auroc_biomarker'],
                'biomarker_metrics_json': json.dumps(result['biomarker_metrics']),
                'num_biomarkers': len(result['biomarker_metrics'])
            }
            
            # Add flag columns
            for flag in sorted(all_flags):
                row[f'flag_{flag}'] = 1 if flag in result['flags'] else 0
            
            writer.writerow(row)
    
    print(f"Results written to: {output_path}")
    print(f"Total models processed: {len(results)}")

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description='Analyze model metrics from Comorbidities-Detection models directory (direct model directories)')
    parser.add_argument('--models_dir', type=str, 
                       default="/lfs/turing1/0/mahmedc/Comorbidities-Detection/models",
                       help='Path to the models directory (default: /lfs/turing1/0/mahmedc/Comorbidities-Detection/models)')
    parser.add_argument('--output_dir', type=str,
                       default="/lfs/turing1/0/mahmedc/Comorbidities-Detection/CT-Disease-Detection",
                       help='Path to the output directory (default: /lfs/turing1/0/mahmedc/Comorbidities-Detection/CT-Disease-Detection)')
    parser.add_argument('--output_filename', type=str,
                       help='Output filename (default: model_metrics_summary_TIMESTAMP.csv)')
    
    args = parser.parse_args()
    
    # Set up paths
    models_dir = args.models_dir
    output_dir = args.output_dir
    
    # Generate timestamp and filename
    if args.output_filename:
        output_filename = args.output_filename
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"model_metrics_direct_{timestamp}.csv"
    
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Starting direct model metrics analysis...")
    print(f"Models directory: {models_dir}")
    print(f"Output file: {output_path}")
    
    # Scan models directory
    results = scan_models_directory(models_dir)
    
    # Write results to CSV
    write_results_to_csv(results, output_path)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()

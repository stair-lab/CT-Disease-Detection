#!/usr/bin/env python3
"""
Script to analyze model metrics from the Comorbidities-Detection models directory.
Extracts training progress information and generates a comprehensive CSV summary with ranking.
Supports both classification (AUROC-based) and regression (MAE-based) tasks.
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
import pandas as pd

def parse_model_name(model_name: str) -> Tuple[str, str]:
    """
    Parse model directory name to extract architecture and learning rate.
    
    Args:
        model_name: Model directory name (e.g., "DenseNet-121_lr1e-03_bs16_20250921_091612")
        
    Returns:
        Tuple of (architecture, learning_rate)
    """
    # Extract architecture (everything before the first underscore)
    parts = model_name.split('_')
    architecture = parts[0]
    
    # Extract learning rate (look for lr pattern)
    lr_match = re.search(r'lr([\d.e-]+)', model_name)
    learning_rate = lr_match.group(1) if lr_match else "unknown"
    
    return architecture, learning_rate

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
        best_epoch_line_idx: Index of the line with the best validation AUROC or MAE
        
    Returns:
        List of dictionaries containing biomarker metrics
    """
    biomarker_metrics = []
    biomarker_dict = {}  # Use dict to deduplicate by biomarker name
    
    # Look for the "Validation metrics per biomarker:" line around the best epoch
    # Search both forward and backward from the best epoch line
    search_range = list(range(best_epoch_line_idx, min(best_epoch_line_idx + 10, len(log_lines)))) + \
                   list(range(best_epoch_line_idx - 1, max(0, best_epoch_line_idx - 10), -1))
    
    for i in search_range:
        if i < 0 or i >= len(log_lines):
            continue
            
        line = log_lines[i]
        
        if "Validation metrics per biomarker:" in line:
            # Parse the next few lines for biomarker metrics
            for j in range(i + 1, min(i + 15, len(log_lines))):
                if j >= len(log_lines):
                    break
                    
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
                
                # Check if this line contains biomarker metrics (contains colon and metric values)
                if ":" in metric_line and ("AUROC=" in metric_line or "MSE=" in metric_line or "MAE=" in metric_line):
                    # Parse biomarker metric line
                    # Format: "2025-09-15 19:44:45,857 -   CALCIUMSCORING_ABDOMINALAGATSTON_BINARY: AUROC=0.8172, Acc=0.7248, F1=0.5449"
                    # or "2025-09-21 09:22:21,070 -   AGE: MSE=106.8339, MAE=7.9983" for regression
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
                            # Improved MAE regex to handle different formats
                            mae_match = re.search(r'MAE=([\d.]+)', metrics_str)
                            if mae_match:
                                metrics['MAE'] = float(mae_match.group(1))
                        
                        if metrics:  # Only add if we found at least one metric
                            # Merge metrics for the same biomarker name
                            if biomarker_name in biomarker_dict:
                                biomarker_dict[biomarker_name].update(metrics)
                            else:
                                biomarker_dict[biomarker_name] = {
                                    'biomarker': biomarker_name,
                                    **metrics
                                }
                            
                    except Exception as e:
                        print(f"Warning: Could not parse biomarker metric line: {metric_line}")
                        continue
            
            # If we found metrics, break out of the search
            if biomarker_dict:
                break
    
    # Convert dict back to list
    biomarker_metrics = list(biomarker_dict.values())
    return biomarker_metrics

def find_best_epoch_metrics(log_file_path: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], List[Dict[str, Any]], bool]:
    """
    Find the best validation metrics and extract corresponding metrics.
    For classification: uses median AUROC as selection criterion
    For regression: uses MAE as selection criterion (lower is better)
    
    Args:
        log_file_path: Path to the training progress log file
        
    Returns:
        Tuple of (best_val_avg_auroc, best_train_avg_auroc, best_val_median_auroc, best_train_median_auroc, biomarker_metrics, is_regression)
    """
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading log file {log_file_path}: {e}")
        return None, None, None, None, [], False
    
    best_val_avg_auroc = 0.0
    best_train_avg_auroc = 0.0
    best_val_median_auroc = 0.0
    best_train_median_auroc = 0.0
    best_epoch_line_idx = -1
    
    # Check if this is a regression-only training (AUROC values are all 0.0000)
    is_regression_only = True
    for line in lines:
        train_avg, val_avg, train_median, val_median = extract_auroc_from_line(line)
        if (train_avg is not None and train_avg > 0.0) or (val_avg is not None and val_avg > 0.0) or \
           (train_median is not None and train_median > 0.0) or (val_median is not None and val_median > 0.0):
            is_regression_only = False
            break
    
    if is_regression_only:
        # For regression: find best MAE (lower is better)
        best_mae = float('inf')
        best_mae_line_idx = -1
        
        # First pass: find the best MAE value by looking for "🎯 New best model! Average MAE:" pattern
        for i in range(len(lines)):
            line = lines[i]
            # Look for the "🎯 New best model! Average MAE:" pattern to find best model
            mae_match = re.search(r'🎯 New best model! Average MAE: ([\d.]+)', line)
            if mae_match:
                current_mae = float(mae_match.group(1))
                if current_mae < best_mae:
                    best_mae = current_mae
                    best_mae_line_idx = i
        
        # Second pass: find the biomarker metrics for the epoch with the best MAE
        # Look backwards from the best MAE line to find the "Validation metrics per biomarker:" line
        for i in range(best_mae_line_idx, max(0, best_mae_line_idx - 20), -1):
            if "Validation metrics per biomarker:" in lines[i]:
                best_epoch_line_idx = i
                break
    else:
        # For classification: find the best validation median AUROC (higher is better)
        # Look for the "🎯 New best model! Median AUROC:" pattern
        for i in range(len(lines)):
            line = lines[i]
            # Look for the "🎯 New best model! Median AUROC:" pattern
            best_model_match = re.search(r'🎯 New best model! Median AUROC: ([\d.]+)', line)
            if best_model_match:
                current_median_auroc = float(best_model_match.group(1))
                if current_median_auroc > best_val_median_auroc:
                    best_val_median_auroc = current_median_auroc
                    
                    # Find the corresponding AUROC metrics by looking backwards
                    for j in range(i, max(0, i - 10), -1):
                        train_avg, val_avg, train_median, val_median = extract_auroc_from_line(lines[j])
                        if val_median is not None and abs(val_median - current_median_auroc) < 0.0001:
                            best_train_median_auroc = train_median if train_median is not None else 0.0
                            best_val_avg_auroc = val_avg if val_avg is not None else 0.0
                            best_train_avg_auroc = train_avg if train_avg is not None else 0.0
                            best_epoch_line_idx = j
                            break
    
    # Extract biomarker metrics for the best epoch
    biomarker_metrics = []
    if best_epoch_line_idx >= 0:
        biomarker_metrics = extract_biomarker_metrics(lines, best_epoch_line_idx)
    
    return best_val_avg_auroc, best_train_avg_auroc, best_val_median_auroc, best_train_median_auroc, biomarker_metrics, is_regression_only

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

def calculate_regression_metrics(biomarker_metrics: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    Calculate average MAE and MSE from biomarker metrics for regression tasks.
    
    Args:
        biomarker_metrics: List of biomarker metric dictionaries
        
    Returns:
        Tuple of (avg_mae, avg_mse, mae_biomarker, mse_biomarker)
    """
    mae_values = []
    mse_values = []
    mae_biomarkers = []
    mse_biomarkers = []
    
    for metric in biomarker_metrics:
        if 'MAE' in metric:
            mae_values.append(metric['MAE'])
            mae_biomarkers.append(metric['biomarker'])
        if 'MSE' in metric:
            mse_values.append(metric['MSE'])
            mse_biomarkers.append(metric['biomarker'])
    
    avg_mae = statistics.mean(mae_values) if mae_values else None
    avg_mse = statistics.mean(mse_values) if mse_values else None
    
    # Find the biomarker with the best (lowest) MAE and MSE
    best_mae_biomarker = None
    best_mse_biomarker = None
    
    if mae_values:
        min_mae_idx = mae_values.index(min(mae_values))
        best_mae_biomarker = mae_biomarkers[min_mae_idx]
    
    if mse_values:
        min_mse_idx = mse_values.index(min(mse_values))
        best_mse_biomarker = mse_biomarkers[min_mse_idx]
    
    return avg_mae, avg_mse, best_mae_biomarker, best_mse_biomarker

def scan_models_directory(models_dir: str, biomarker_name: str = "direct") -> List[Dict[str, Any]]:
    """
    Scan the models directory and extract metrics from all model runs.
    
    Args:
        models_dir: Path to the models directory
        biomarker_name: Name of the biomarker (from directory name)
        
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
                val_avg_auroc, train_avg_auroc, val_median_auroc, train_median_auroc, biomarker_metrics, is_regression = find_best_epoch_metrics(log_file)
                
                # Parse model name to extract architecture and learning rate
                architecture, learning_rate = parse_model_name(model_dir)
                
                if val_median_auroc is not None or biomarker_metrics:
                    # Calculate classification metrics
                    median_auroc, median_biomarker = calculate_median_auroc(biomarker_metrics)
                    
                    # Calculate regression metrics
                    avg_mae, avg_mse, best_mae_biomarker, best_mse_biomarker = calculate_regression_metrics(biomarker_metrics)
                    
                    result = {
                        'biomarker': biomarker_name,
                        'architecture': architecture,
                        'learning_rate': learning_rate,
                        'model_name': model_dir,
                        'best_val_avg_auroc': val_avg_auroc,
                        'best_train_avg_auroc': train_avg_auroc,
                        'best_val_median_auroc': val_median_auroc,
                        'best_train_median_auroc': train_median_auroc,
                        'biomarker_median_auroc': median_auroc,
                        'biomarker_median_auroc_biomarker': median_biomarker,
                        'avg_mae': avg_mae,
                        'avg_mse': avg_mse,
                        'best_mae_biomarker': best_mae_biomarker,
                        'best_mse_biomarker': best_mse_biomarker,
                        'is_regression': is_regression,
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
                    val_avg_auroc, train_avg_auroc, val_median_auroc, train_median_auroc, biomarker_metrics, is_regression = find_best_epoch_metrics(log_file)
                    
                    if val_median_auroc is not None or biomarker_metrics:
                        # Calculate classification metrics
                        median_auroc, median_biomarker = calculate_median_auroc(biomarker_metrics)
                        
                        # Calculate regression metrics
                        avg_mae, avg_mse, best_mae_biomarker, best_mse_biomarker = calculate_regression_metrics(biomarker_metrics)
                        
                        # Parse model name to extract architecture and learning rate
                        architecture, learning_rate = parse_model_name(model_dir)
                        
                        result = {
                            'biomarker': config_type,
                            'architecture': architecture,
                            'learning_rate': learning_rate,
                            'model_name': model_dir,
                            'best_val_avg_auroc': val_avg_auroc,
                            'best_train_avg_auroc': train_avg_auroc,
                            'best_val_median_auroc': val_median_auroc,
                            'best_train_median_auroc': train_median_auroc,
                            'biomarker_median_auroc': median_auroc,
                            'biomarker_median_auroc_biomarker': median_biomarker,
                            'avg_mae': avg_mae,
                            'avg_mse': avg_mse,
                            'best_mae_biomarker': best_mae_biomarker,
                            'best_mse_biomarker': best_mse_biomarker,
                            'is_regression': is_regression,
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
                                val_avg_auroc, train_avg_auroc, val_median_auroc, train_median_auroc, biomarker_metrics, is_regression = find_best_epoch_metrics(log_file)
                                
                                if val_median_auroc is not None or biomarker_metrics:
                                    # Calculate classification metrics
                                    median_auroc, median_biomarker = calculate_median_auroc(biomarker_metrics)
                                    
                                    # Calculate regression metrics
                                    avg_mae, avg_mse, best_mae_biomarker, best_mse_biomarker = calculate_regression_metrics(biomarker_metrics)
                                    
                                    # Parse model name to extract architecture and learning rate
                                    architecture, learning_rate = parse_model_name(model_dir)
                                    
                                    result = {
                                        'biomarker': config_type,
                                        'architecture': architecture,
                                        'learning_rate': learning_rate,
                                        'model_name': model_dir,
                                        'best_val_avg_auroc': val_avg_auroc,
                                        'best_train_avg_auroc': train_avg_auroc,
                                        'best_val_median_auroc': val_median_auroc,
                                        'best_train_median_auroc': train_median_auroc,
                                        'biomarker_median_auroc': median_auroc,
                                        'biomarker_median_auroc_biomarker': median_biomarker,
                                        'avg_mae': avg_mae,
                                        'avg_mse': avg_mse,
                                        'best_mae_biomarker': best_mae_biomarker,
                                        'best_mse_biomarker': best_mse_biomarker,
                                        'is_regression': is_regression,
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
                            val_avg_auroc, train_avg_auroc, val_median_auroc, train_median_auroc, biomarker_metrics, is_regression = find_best_epoch_metrics(log_file)
                            
                            if val_median_auroc is not None or biomarker_metrics:
                                # Calculate classification metrics
                                median_auroc, median_biomarker = calculate_median_auroc(biomarker_metrics)
                                
                                # Calculate regression metrics
                                avg_mae, avg_mse, best_mae_biomarker, best_mse_biomarker = calculate_regression_metrics(biomarker_metrics)
                                
                                # Parse model name to extract architecture and learning rate
                                architecture, learning_rate = parse_model_name(model_dir)
                                
                                result = {
                                    'biomarker': config_type,
                                    'architecture': architecture,
                                    'learning_rate': learning_rate,
                                    'model_name': model_dir,
                                    'best_val_avg_auroc': val_avg_auroc,
                                    'best_train_avg_auroc': train_avg_auroc,
                                    'best_val_median_auroc': val_median_auroc,
                                    'best_train_median_auroc': train_median_auroc,
                                    'biomarker_median_auroc': median_auroc,
                                    'biomarker_median_auroc_biomarker': median_biomarker,
                                    'avg_mae': avg_mae,
                                    'avg_mse': avg_mse,
                                    'best_mae_biomarker': best_mae_biomarker,
                                    'best_mse_biomarker': best_mse_biomarker,
                                    'is_regression': is_regression,
                                    'biomarker_metrics': biomarker_metrics,
                                    'flags': [subdir]
                                }
                                results.append(result)
    
    return results

def write_results_to_csv(results: List[Dict[str, Any]], output_path: str):
    """
    Write results to CSV file, sorted by biomarker, architecture, and rank.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output CSV file
    """
    if not results:
        print("No results to write.")
        return
    
    # Sort results by biomarker, architecture, and rank (ascending)
    sorted_results = sorted(results, key=lambda x: (
        x.get('biomarker', ''),
        x.get('architecture', ''),
        x.get('rank', float('inf'))  # Use infinity for missing ranks so they appear last
    ))
    
    # Get all possible flag columns
    all_flags = set()
    for result in sorted_results:
        all_flags.update(result['flags'])
    
    # Define CSV columns
    columns = [
        'biomarker',
        'architecture',
        'learning_rate',
        'model_name',
        'rank',
        'best_val_avg_auroc',
        'best_train_avg_auroc',
        'best_val_median_auroc',
        'best_train_median_auroc',
        'biomarker_median_auroc',
        'biomarker_median_auroc_biomarker',
        'avg_mae',
        'avg_mse',
        'best_mae_biomarker',
        'best_mse_biomarker',
        'is_regression',
        'biomarker_metrics_json',
        'num_biomarkers'
    ]
    
    # Add flag columns
    for flag in sorted(all_flags):
        columns.append(f'flag_{flag}')
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        
        for result in sorted_results:
            row = {
                'biomarker': result['biomarker'],
                'architecture': result['architecture'],
                'learning_rate': result['learning_rate'],
                'model_name': result['model_name'],
                'rank': result['rank'],
                'best_val_avg_auroc': result['best_val_avg_auroc'],
                'best_train_avg_auroc': result['best_train_avg_auroc'],
                'best_val_median_auroc': result['best_val_median_auroc'],
                'best_train_median_auroc': result['best_train_median_auroc'],
                'biomarker_median_auroc': result['biomarker_median_auroc'],
                'biomarker_median_auroc_biomarker': result['biomarker_median_auroc_biomarker'],
                'avg_mae': result['avg_mae'],
                'avg_mse': result['avg_mse'],
                'best_mae_biomarker': result['best_mae_biomarker'],
                'best_mse_biomarker': result['best_mse_biomarker'],
                'is_regression': result['is_regression'],
                'biomarker_metrics_json': json.dumps(result['biomarker_metrics']),
                'num_biomarkers': len(result['biomarker_metrics'])
            }
            
            # Add flag columns
            for flag in sorted(all_flags):
                row[f'flag_{flag}'] = 1 if flag in result['flags'] else 0
            
            writer.writerow(row)
    
    print(f"Results written to: {output_path}")
    print(f"Total models processed: {len(sorted_results)}")

def add_ranking_to_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add ranking based on biomarker and architecture groups.
    For classification: rank by Val Median AUROC (higher is better)
    For regression: rank by avg_mae (lower is better)
    
    Args:
        results: List of result dictionaries
        
    Returns:
        List of result dictionaries with ranking added
    """
    if not results:
        return results
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Add ranking column
    df['rank'] = 0
    
    # Group by biomarker and architecture
    for (biomarker, architecture), group in df.groupby(['biomarker', 'architecture']):
        if len(group) > 1:  # Only rank if there are multiple models
            # Check if this group has regression models
            has_regression = group['is_regression'].any()
            
            if has_regression:
                # For regression: rank by avg_mae (lower is better)
                # Handle NaN values by putting them at the end
                group_sorted = group.sort_values('avg_mae', na_position='last')
                group_sorted['rank'] = range(1, len(group_sorted) + 1)
            else:
                # For classification: rank by best_val_median_auroc (higher is better)
                group_sorted = group.sort_values('best_val_median_auroc', ascending=False, na_position='last')
                group_sorted['rank'] = range(1, len(group_sorted) + 1)
            
            # Update the original dataframe
            df.loc[group_sorted.index, 'rank'] = group_sorted['rank']
        else:
            # Single model gets rank 1
            df.loc[group.index, 'rank'] = 1
    
    # Convert back to list of dictionaries
    return df.to_dict('records')

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description='Analyze model metrics from Comorbidities-Detection models directory with ranking')
    parser.add_argument('--models_base_dir', type=str, 
                       default="/lfs/turing1/0/mahmedc/Comorbidities-Detection/models",
                       help='Base path to the models directory (default: /lfs/turing1/0/mahmedc/Comorbidities-Detection/models)')
    parser.add_argument('--directories', type=str, nargs='+',
                       help='List of directory names to process (e.g., age_only calcium_only). If not provided, will auto-detect directories with "only" in name.')
    parser.add_argument('--output_dir', type=str,
                       default="/lfs/turing1/0/mahmedc/Comorbidities-Detection/CT-Disease-Detection",
                       help='Path to the output directory (default: /lfs/turing1/0/mahmedc/Comorbidities-Detection/CT-Disease-Detection)')
    parser.add_argument('--output_filename', type=str,
                       help='Output filename (default: model_metrics_with_ranking_TIMESTAMP.csv)')
    
    args = parser.parse_args()
    
    # Set up paths
    models_base_dir = args.models_base_dir
    output_dir = args.output_dir
    
    # Determine which directories to process
    if args.directories:
        directories_to_process = args.directories
    else:
        # Auto-detect directories with "only" in the name
        all_dirs = [d for d in os.listdir(models_base_dir) 
                   if os.path.isdir(os.path.join(models_base_dir, d)) 
                   and not d.startswith('.')]
        directories_to_process = [d for d in all_dirs if "only" in d]
    
    print(f"Directories to process: {directories_to_process}")
    
    # Generate timestamp and filename
    if args.output_filename:
        output_filename = args.output_filename
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"model_metrics_with_ranking_{timestamp}.csv"
    
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Starting model metrics analysis with ranking...")
    print(f"Models base directory: {models_base_dir}")
    print(f"Output file: {output_path}")
    
    # Collect results from all directories
    all_results = []
    for directory in directories_to_process:
        directory_path = os.path.join(models_base_dir, directory)
        if os.path.exists(directory_path):
            print(f"\nProcessing directory: {directory}")
            results = scan_models_directory(directory_path, biomarker_name=directory)
            all_results.extend(results)
        else:
            print(f"Warning: Directory {directory_path} does not exist, skipping...")
    
    # Add ranking to results
    print(f"\nAdding ranking to {len(all_results)} results...")
    all_results = add_ranking_to_results(all_results)
    
    # Write results to CSV
    write_results_to_csv(all_results, output_path)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()

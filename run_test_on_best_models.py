#!/usr/bin/env python3
"""
Script to run test.py on all rank=1 models from validation results
and compile test set metrics into a comprehensive CSV file.
"""

import pandas as pd
import subprocess
import os
import json
import sys
import re
from pathlib import Path
import time
from typing import Dict, Any, List

def load_best_models(csv_path: str, architecture_filter: str = None) -> pd.DataFrame:
    """Load and filter to rank=1 models"""
    df = pd.read_csv(csv_path)
    best_models = df[df['rank'] == 1].copy()
    if architecture_filter:
        best_models = best_models[best_models['architecture'] == architecture_filter]
        print(f"Filtering to architecture: {architecture_filter}")
    print(f"Found {len(best_models)} best models to test")
    return best_models

def construct_checkpoint_path(model_name: str, biomarker: str) -> str:
    """Construct the full path to the checkpoint"""
    base_path = "/lfs/skampere2/0/mahmedc/Comorbidities-Detection/models"
    return os.path.join(base_path, biomarker, model_name, "best_checkpoint.pth")

def construct_biomarker_config_path(biomarker: str) -> str:
    """Construct the path to the biomarker config file"""
    # Remove "_only" suffix from biomarker name for config file
    biomarker_base = biomarker.replace("_only", "")
    config_path = f"config/biomarker_config_{biomarker_base}.yaml"
    
    # Check if YAML exists, if not try JSON
    if not os.path.exists(config_path):
        json_path = f"config/biomarker_config_{biomarker_base}.json"
        if os.path.exists(json_path):
            return json_path
    
    return config_path

def run_test_on_model(model_name: str, biomarker: str, data_dir: str, output_base_dir: str, test_csv: str = "test.csv") -> Dict[str, Any]:
    """Run test.py on a single model and return the results"""
    
    # Construct paths
    checkpoint_path = construct_checkpoint_path(model_name, biomarker)
    biomarker_config_path = construct_biomarker_config_path(biomarker)
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return {"error": "Checkpoint not found"}
    
    # Check if biomarker config exists
    if not os.path.exists(biomarker_config_path):
        print(f"❌ Biomarker config not found: {biomarker_config_path}")
        return {"error": "Biomarker config not found"}
    
    # Create output directory (even though we're not using it currently)
    output_dir = os.path.join(output_base_dir, f"{biomarker}_{model_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct test.py command
    cmd = [
        "python", "test.py",
        "--data_dir", data_dir,
        "--checkpoint_path", checkpoint_path,
        "--biomarker_config", biomarker_config_path,
        "--test_csv", test_csv
    ]
    
    # Optional: Add output_dir if you want individual result directories
    # cmd.extend(["--output_dir", output_dir])
    
    print(f"🧪 Validating {biomarker} - {model_name} (using {test_csv})")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Config: {biomarker_config_path}")
    
    try:
        # Set TORCH_HOME to prevent permission errors when downloading model weights
        env = os.environ.copy()
        env['TORCH_HOME'] = '/lfs/skampere2/0/mahmedc/.cache/torch'
        
        # Run the test
        result = subprocess.run(cmd, capture_output=True, text=True, 
                               cwd="/lfs/skampere2/0/mahmedc/Comorbidities-Detection/CT-Disease-Detection",
                               env=env)
        
        if result.returncode != 0:
            print(f"❌ Validation failed for {biomarker} - {model_name}")
            print(f"   Error: {result.stderr}")
            # Also print stdout in case there are useful error messages there
            if result.stdout:
                print(f"   Stdout: {result.stdout[-500:]}")  # Last 500 chars
            return {"error": f"Validation failed: {result.stderr}"}
        
        # Parse metrics from stdout (test.py prints the results to console)
        print(f"✅ Validation completed for {biomarker} - {model_name}")
        
        # For now, return success status - we'll need to parse metrics from stdout
        # or modify test.py to return metrics in a different way
        return {
            "status": "success",
            "stdout": result.stdout,
            "output_dir": output_dir
        }
            
    except Exception as e:
        print(f"❌ Exception during test for {biomarker} - {model_name}: {str(e)}")
        return {"error": f"Exception: {str(e)}"}

def parse_metrics_from_stdout(stdout: str, biomarker: str) -> Dict[str, Any]:
    """Parse metrics from test.py stdout output"""
    metrics = {}
    
    lines = stdout.split('\n')
    
    # Look for regression metrics (for age_only)
    if biomarker == "age_only":
        for line in lines:
            if "MAE:" in line:
                # Extract MAE value and CI
                mae_match = re.search(r'MAE:\s+([\d.]+)\s+\[([\d.]+),\s+([\d.]+)\]', line)
                if mae_match:
                    try:
                        metrics["mae"] = float(mae_match.group(1))
                        metrics["mae_ci"] = [float(mae_match.group(2)), float(mae_match.group(3))]
                    except ValueError:
                        print(f"   ⚠️  Warning: Could not parse MAE values from line: {line.strip()}")
            
            elif "MSE:" in line:
                # Extract MSE value and CI
                mse_match = re.search(r'MSE:\s+([\d.]+)\s+\[([\d.]+),\s+([\d.]+)\]', line)
                if mse_match:
                    try:
                        metrics["mse"] = float(mse_match.group(1))
                        metrics["mse_ci"] = [float(mse_match.group(2)), float(mse_match.group(3))]
                    except ValueError:
                        print(f"   ⚠️  Warning: Could not parse MSE values from line: {line.strip()}")
            
            elif "R²:" in line:
                # Extract R² value and CI (handle negative values and scientific notation)
                r2_match = re.search(r'R²:\s+([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s+\[([-+]?[\d.]+(?:[eE][-+]?\d+)?),\s+([-+]?[\d.]+(?:[eE][-+]?\d+)?)\]', line)
                if r2_match:
                    try:
                        metrics["r2_score"] = float(r2_match.group(1))
                        metrics["r2_score_ci"] = [float(r2_match.group(2)), float(r2_match.group(3))]
                    except ValueError as e:
                        print(f"   ⚠️  Warning: Could not parse R² values from line: {line.strip()}")
    
    # Look for classification metrics (for binary tasks)
    else:
        for line in lines:
            if "AUROC:" in line:
                auroc_match = re.search(r'AUROC:\s+([\d.]+)\s+\[([\d.]+),\s+([\d.]+)\]', line)
                if auroc_match:
                    metrics["auroc"] = float(auroc_match.group(1))
                    metrics["auroc_ci"] = [float(auroc_match.group(2)), float(auroc_match.group(3))]
            
            elif "Precision:" in line:
                precision_match = re.search(r'Precision:\s+([\d.]+)\s+\[([\d.]+),\s+([\d.]+)\]', line)
                if precision_match:
                    metrics["precision"] = float(precision_match.group(1))
                    metrics["precision_ci"] = [float(precision_match.group(2)), float(precision_match.group(3))]
            
            elif "Recall:" in line:
                recall_match = re.search(r'Recall:\s+([\d.]+)\s+\[([\d.]+),\s+([\d.]+)\]', line)
                if recall_match:
                    metrics["recall"] = float(recall_match.group(1))
                    metrics["recall_ci"] = [float(recall_match.group(2)), float(recall_match.group(3))]
            
            elif "Specificity:" in line:
                specificity_match = re.search(r'Specificity:\s+([\d.]+)\s+\[([\d.]+),\s+([\d.]+)\]', line)
                if specificity_match:
                    metrics["specificity"] = float(specificity_match.group(1))
                    metrics["specificity_ci"] = [float(specificity_match.group(2)), float(specificity_match.group(3))]
            
            elif "F1-Score:" in line:
                f1_match = re.search(r'F1-Score:\s+([\d.]+)\s+\[([\d.]+),\s+([\d.]+)\]', line)
                if f1_match:
                    metrics["f1_score"] = float(f1_match.group(1))
                    metrics["f1_score_ci"] = [float(f1_match.group(2)), float(f1_match.group(3))]
            
            elif "Accuracy:" in line:
                accuracy_match = re.search(r'Accuracy:\s+([\d.]+)\s+\[([\d.]+),\s+([\d.]+)\]', line)
                if accuracy_match:
                    metrics["accuracy"] = float(accuracy_match.group(1))
                    metrics["accuracy_ci"] = [float(accuracy_match.group(2)), float(accuracy_match.group(3))]
            
            # Look for threshold information
            elif "Threshold used:" in line:
                threshold_match = re.search(r'Threshold used:\s+([\d.]+)', line)
                if threshold_match:
                    metrics["threshold_used"] = float(threshold_match.group(1))
            
            elif "Optimal threshold:" in line:
                threshold_match = re.search(r'Optimal threshold:\s+([\d.]+)', line)
                if threshold_match:
                    metrics["optimal_threshold"] = float(threshold_match.group(1))
    
    return metrics

def extract_metrics_from_result(result: Dict[str, Any], biomarker: str, architecture: str, learning_rate: str, model_name: str) -> Dict[str, Any]:
    """Extract and format metrics from test result"""
    
    if "error" in result:
        return {
            "biomarker": biomarker,
            "architecture": architecture,
            "learning_rate": learning_rate,
            "model_name": model_name,
            "status": "failed",
            "error": result["error"]
        }
    
    # Parse metrics from stdout
    stdout = result.get("stdout", "")
    metrics = parse_metrics_from_stdout(stdout, biomarker)
    
    # Base information
    row = {
        "biomarker": biomarker,
        "architecture": architecture,
        "learning_rate": learning_rate,
        "model_name": model_name,
        "status": "success",
        "output_dir": result.get("output_dir", "")
    }
    
    # Add all parsed metrics to the row
    for metric_name, value in metrics.items():
        if metric_name.endswith("_ci"):
            continue  # Skip CI values, we'll handle them separately
        row[f"test_{metric_name}"] = value
        
        # Add CI if available
        if f"{metric_name}_ci" in metrics:
            ci = metrics[f"{metric_name}_ci"]
            if isinstance(ci, list) and len(ci) == 2:
                row[f"test_{metric_name}_ci_lower"] = ci[0]
                row[f"test_{metric_name}_ci_upper"] = ci[1]
    
    return row

def main():
    """Main function to run tests on all best models"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run test.py on best models from validation results')
    parser.add_argument('--validation_results_csv', type=str,
                       default="/lfs/skampere2/0/mahmedc/Comorbidities-Detection/CT-Disease-Detection/model_metrics_with_ranking_20251117_103355.csv",
                       help='Path to CSV file with validation results and rankings')
    parser.add_argument('--data_dir', type=str,
                       default="/lfs/skampere2/0/mahmedc/Comorbidities-Detection/datasets/full_data",
                       help='Path to data directory')
    parser.add_argument('--output_base_dir', type=str,
                       default="test_results_best_models",
                       help='Base directory for test results')
    parser.add_argument('--results_csv', type=str,
                       default=None,
                       help='Output CSV file for test results (default: auto-generated)')
    parser.add_argument('--architecture', type=str,
                       default=None,
                       help='Filter to specific architecture (e.g., "Swin", "ViT-Small"). If not provided, tests all rank=1 models.')
    parser.add_argument('--test_csv', type=str,
                       default="test.csv",
                       help='Test CSV file to use (default: test.csv)')
    parser.add_argument('--run_on_val', action='store_true',
                       help='Run on validation set (val.csv) instead of test set. Useful for verifying test.py replicates training validation metrics.')
    
    args = parser.parse_args()
    
    # If --run_on_val is set, override test_csv to val.csv
    if args.run_on_val:
        args.test_csv = "val.csv"
        if args.output_base_dir == "test_results_best_models":
            args.output_base_dir = "val_results_best_models"
    
    # Generate default results CSV if not provided
    if args.results_csv is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arch_suffix = f"_{args.architecture}" if args.architecture else ""
        dataset_suffix = "_val" if args.run_on_val else "_test"
        args.results_csv = f"single_biomarker{dataset_suffix}_results_{timestamp}{arch_suffix}.csv"
    
    dataset_name = "validation set" if args.run_on_val else "test set"
    print(f"🚀 Starting {dataset_name} inference on best models (using {args.test_csv})")
    print("=" * 60)
    print(f"Validation results CSV: {args.validation_results_csv}")
    print(f"Architecture filter: {args.architecture if args.architecture else 'None (all architectures)'}")
    print(f"CSV file: {args.test_csv}")
    print(f"Output CSV: {args.results_csv}")
    print("=" * 60)
    
    # Load best models
    best_models = load_best_models(args.validation_results_csv, architecture_filter=args.architecture)
    
    # Create output directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Results storage
    all_results = []
    
    # Process each model
    for i, (idx, row) in enumerate(best_models.iterrows()):
        biomarker = row['biomarker']
        architecture = row['architecture']
        learning_rate = row['learning_rate']
        model_name = row['model_name']
        
        print(f"\n[{i+1}/{len(best_models)}] Processing: {biomarker} - {architecture} - {learning_rate}")
        
        # Run test
        result = run_test_on_model(model_name, biomarker, args.data_dir, args.output_base_dir, test_csv=args.test_csv)
        
        # Extract metrics
        metrics_row = extract_metrics_from_result(result, biomarker, architecture, learning_rate, model_name)
        all_results.append(metrics_row)
        
        # Small delay to avoid overwhelming the system
        time.sleep(1)
    
    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(args.results_csv, index=False)
    else:
        print("⚠️  No results to save!")
        return
    
    dataset_name = "validation set" if args.run_on_val else "test set"
    print(f"\n🎉 All {dataset_name} inferences completed!")
    print(f"📊 Results saved to: {args.results_csv}")
    print(f"📁 Individual results in: {args.output_base_dir}/")
    if args.run_on_val:
        print(f"\n💡 Compare these validation results with validation metrics from training logs!")
        print(f"   This verifies that test.py can replicate the training validation performance.")
    else:
        print(f"\n💡 Compare these test results with validation metrics from training!")
    
    # Summary statistics
    successful_tests = results_df[results_df['status'] == 'success']
    failed_tests = results_df[results_df['status'] == 'failed']
    
    print(f"\n📈 Summary:")
    print(f"   ✅ Successful {dataset_name} runs: {len(successful_tests)}")
    print(f"   ❌ Failed {dataset_name} runs: {len(failed_tests)}")
    
    if len(failed_tests) > 0:
        print(f"\n❌ Failed {dataset_name} runs:")
        for _, row in failed_tests.iterrows():
            print(f"   - {row['biomarker']} - {row['architecture']}: {row.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()

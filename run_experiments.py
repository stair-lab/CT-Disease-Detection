#!/usr/bin/env python3
"""
Experiment Runner for Multi-Task Comorbidity Detection
Runs experiments based on the CSV configuration file
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.experiment_config import ExperimentConfigLoader
from model.model_factory import get_model_memory_requirement


def check_gpu_memory():
    """Check available GPU memory"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', 
                               '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for i, line in enumerate(lines):
                total, used, free = map(int, line.split(', '))
                gpu_info.append({
                    'gpu_id': i,
                    'total_mb': total,
                    'used_mb': used,
                    'free_mb': free,
                    'free_gb': free / 1024
                })
            return gpu_info
        else:
            print("Warning: Could not check GPU memory")
            return []
    except FileNotFoundError:
        print("Warning: nvidia-smi not found. Cannot check GPU memory.")
        return []


def estimate_memory_needed(memory_requirement_str):
    """Estimate memory needed from requirement string"""
    # Parse strings like "12-16GB" or "4-6GB"
    if '-' in memory_requirement_str and 'GB' in memory_requirement_str:
        try:
            # Extract the upper bound
            upper_bound = memory_requirement_str.split('-')[1].replace('GB', '').strip()
            return float(upper_bound)
        except:
            pass
    
    # Fallback estimates
    memory_estimates = {
        '4-6GB': 6,
        '6-8GB': 8,
        '8-10GB': 10,
        '10-12GB': 12,
        '12-16GB': 16,
        '14-18GB': 18,
        '16-20GB': 20,
        '20-24GB': 24,
        '24-32GB': 32
    }
    
    return memory_estimates.get(memory_requirement_str, 16)  # Default 16GB


def can_run_experiment(config, gpu_info):
    """Check if experiment can run on available GPUs"""
    if not gpu_info:
        return True, 0  # Assume can run if we can't check
    
    memory_needed_gb = estimate_memory_needed(config.expected_gpu_memory)
    
    for gpu in gpu_info:
        if gpu['free_gb'] >= memory_needed_gb:
            return True, gpu['gpu_id']
    
    return False, -1


def run_single_experiment(config, data_dir, output_base_dir, epochs, gpu_id=None):
    """Run a single experiment"""
    
    # Prepare command
    cmd = [
        sys.executable, 'train_enhanced.py',
        '--config_csv', config.csv_path if hasattr(config, 'csv_path') else 'experimentation_plan_simplified.csv',
        '--data_dir', data_dir,
        '--output_base_dir', output_base_dir,
        '--epochs', str(epochs),
        '--model_name', config.model
    ]
    
    # Set GPU environment variable if specified
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"Running command: {' '.join(cmd)}")
    if gpu_id is not None:
        print(f"Using GPU {gpu_id}")
    
    # Run experiment
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Experiment {config.experiment_name} completed successfully")
            return True, result.stdout
        else:
            print(f"❌ Experiment {config.experiment_name} failed")
            print(f"Error: {result.stderr}")
            return False, result.stderr
            
    except Exception as e:
        print(f"❌ Error running experiment {config.experiment_name}: {str(e)}")
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Run Multi-Task Comorbidity Detection Experiments')
    parser.add_argument('--config_csv', 
                       default='experimentation_plan_simplified.csv',
                       help='Path to experiment configuration CSV')
    parser.add_argument('--data_dir', 
                       required=True,
                       help='Path to dataset directory containing train.csv and test.csv')
    parser.add_argument('--output_base_dir', 
                       default='/lfs/turing1/0/mahmedc/Comorbidities-Detection/models',
                       help='Base directory for model outputs')
    parser.add_argument('--epochs', 
                       type=int, 
                       default=100, 
                       help='Number of training epochs')
    parser.add_argument('--model_name', 
                       help='Specific model to train (if not provided, trains all must-include models)')
    parser.add_argument('--must_include_only', 
                       action='store_true',
                       help='Run only must-include experiments')
    parser.add_argument('--dry_run', 
                       action='store_true',
                       help='Show what would be run without actually running')
    parser.add_argument('--check_memory', 
                       action='store_true',
                       help='Check GPU memory requirements before running')
    parser.add_argument('--sequential', 
                       action='store_true',
                       help='Run experiments sequentially instead of trying to optimize GPU usage')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.config_csv):
        print(f"❌ Configuration file not found: {args.config_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Load experiment configurations
    config_loader = ExperimentConfigLoader(args.config_csv)
    
    if args.model_name:
        # Run specific model
        config = config_loader.load_config_by_model(args.model_name)
        if config is None:
            print(f"❌ Model {args.model_name} not found in configuration file")
            available_models = config_loader.get_available_models()
            print(f"Available models: {', '.join(available_models)}")
            sys.exit(1)
        configs_to_run = [config]
    else:
        # Run all or must-include models
        if args.must_include_only:
            configs_to_run = config_loader.load_must_include_configs()
        else:
            configs_to_run = config_loader.load_all_configs()
    
    print(f"📊 Found {len(configs_to_run)} experiments to run")
    
    # Check GPU memory if requested
    gpu_info = []
    if args.check_memory:
        print("🔍 Checking GPU memory...")
        gpu_info = check_gpu_memory()
        if gpu_info:
            print("Available GPUs:")
            for gpu in gpu_info:
                print(f"  GPU {gpu['gpu_id']}: {gpu['free_gb']:.1f}GB free / {gpu['total_mb']/1024:.1f}GB total")
        else:
            print("⚠️  Could not check GPU memory")
    
    # Filter experiments by memory requirements if checking memory
    if args.check_memory and gpu_info:
        runnable_configs = []
        skipped_configs = []
        
        for config in configs_to_run:
            can_run, gpu_id = can_run_experiment(config, gpu_info)
            if can_run:
                runnable_configs.append((config, gpu_id))
            else:
                skipped_configs.append(config)
        
        if skipped_configs:
            print(f"⚠️  Skipping {len(skipped_configs)} experiments due to insufficient GPU memory:")
            for config in skipped_configs:
                print(f"    {config.model} (needs {config.expected_gpu_memory})")
        
        configs_to_run = [config for config, _ in runnable_configs]
        gpu_assignments = {config.model: gpu_id for config, gpu_id in runnable_configs}
    else:
        gpu_assignments = {}
    
    # Show experiment plan
    print(f"\n📋 Experiment Plan:")
    print("=" * 80)
    for i, config in enumerate(configs_to_run):
        gpu_str = f" (GPU {gpu_assignments.get(config.model, 'auto')})" if args.check_memory else ""
        output_dir = os.path.join(args.output_base_dir, config.experiment_name)
        exists_str = " ⚠️ EXISTS" if os.path.exists(output_dir) else ""
        print(f"{i+1:2d}. {config.model:<40} | {config.expected_gpu_memory:<10} | LR: {config.learning_rate}{gpu_str}{exists_str}")
        print(f"    Output: {config.experiment_name}")
    print("=" * 80)
    
    # Check for existing directories
    existing_dirs = []
    for config in configs_to_run:
        output_dir = os.path.join(args.output_base_dir, config.experiment_name)
        if os.path.exists(output_dir):
            existing_dirs.append(config.experiment_name)
    
    if existing_dirs:
        print(f"\n⚠️  Note: {len(existing_dirs)} experiment directories already exist:")
        for dir_name in existing_dirs[:5]:  # Show first 5
            print(f"    {dir_name}")
        if len(existing_dirs) > 5:
            print(f"    ... and {len(existing_dirs) - 5} more")
        print("   The system will automatically create new timestamped directories if needed.")
    
    if args.dry_run:
        print("🔍 Dry run completed. Use --dry_run=false to actually run experiments.")
        return
    
    # Confirm before running
    if not args.model_name:  # Only ask for confirmation when running multiple experiments
        response = input(f"\n🚀 Ready to run {len(configs_to_run)} experiments. Continue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Aborted by user")
            return
    
    # Run experiments
    print(f"\n🏃 Starting experiments...")
    successful_runs = []
    failed_runs = []
    
    for i, config in enumerate(configs_to_run):
        print(f"\n{'='*60}")
        print(f"🔄 Running experiment {i+1}/{len(configs_to_run)}: {config.model}")
        print(f"{'='*60}")
        
        # Determine GPU to use
        gpu_id = gpu_assignments.get(config.model) if args.check_memory else None
        
        # Add csv_path attribute for the run_single_experiment function
        config.csv_path = args.config_csv
        
        success, output = run_single_experiment(
            config=config,
            data_dir=args.data_dir,
            output_base_dir=args.output_base_dir,
            epochs=args.epochs,
            gpu_id=gpu_id
        )
        
        if success:
            successful_runs.append(config.model)
        else:
            failed_runs.append((config.model, output))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Experiment Summary")
    print(f"{'='*60}")
    print(f"✅ Successful: {len(successful_runs)}")
    print(f"❌ Failed: {len(failed_runs)}")
    
    if successful_runs:
        print(f"\n✅ Successful experiments:")
        for model in successful_runs:
            print(f"    {model}")
    
    if failed_runs:
        print(f"\n❌ Failed experiments:")
        for model, error in failed_runs:
            print(f"    {model}: {error[:100]}...")
    
    print(f"\n📁 Results saved to: {args.output_base_dir}")
    print(f"📈 View tensorboard logs: tensorboard --logdir {args.output_base_dir}")


if __name__ == "__main__":
    main()

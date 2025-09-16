#!/usr/bin/env python3
"""
Flexible Experiment Runner for Multi-Task Comorbidity Detection
Runs experiments using the flexible biomarker configuration system
"""

import os
import sys
import argparse
import subprocess
import logging
import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.experiment_config import ExperimentConfigLoader
from config.biomarker_config import FlexibleBiomarkerConfig
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


def is_experiment_completed(config, output_base_dir):
    """Check if experiment is already completed"""
    output_dir = os.path.join(output_base_dir, config.experiment_name)
    
    # Check for completion markers
    completion_markers = [
        os.path.join(output_dir, 'training_complete.txt'),
        os.path.join(output_dir, 'best_model.pth'),
        os.path.join(output_dir, 'final_results.json')
    ]
    
    # If any completion marker exists, consider it completed
    for marker in completion_markers:
        if os.path.exists(marker):
            return True, marker
    
    return False, None


def run_single_experiment(config, data_dir, biomarker_config_path, output_base_dir, epochs, gpu_id=None):
    """Run a single experiment"""
    
    # Ensure Hugging Face cache directory exists on /lfs filesystem
    # Use /lfs path instead of /afs to avoid subprocess permission issues
    hf_cache_dir = '/lfs/turing1/0/mahmedc/.cache/huggingface'
    os.makedirs(os.path.join(hf_cache_dir, 'transformers'), exist_ok=True)
    os.makedirs(os.path.join(hf_cache_dir, 'datasets'), exist_ok=True)
    os.makedirs(os.path.join(hf_cache_dir, 'hub'), exist_ok=True)
    
    # Prepare command
    cmd = [
        sys.executable, 'train.py',
        '--config_csv', config.csv_path if hasattr(config, 'csv_path') else 'experimentation_plan_simplified.csv',
        '--data_dir', data_dir,
        '--biomarker_config', biomarker_config_path,
        '--output_base_dir', output_base_dir,
        '--epochs', str(epochs),
        '--model_name', config.model,
        '--experiment_name', config.experiment_name
    ]
    
    # Add learning rate if this is a single-LR experiment
    if len(config.learning_rate) == 1:
        cmd.extend(['--learning_rate', str(config.learning_rate[0])])
    
    # Set GPU environment variable if specified
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Fix Hugging Face cache directory to avoid AFS permission issues
    # Set cache to a location on /lfs filesystem instead of /afs
    env['HF_HOME'] = hf_cache_dir
    env['TRANSFORMERS_CACHE'] = os.path.join(hf_cache_dir, 'transformers')
    env['HF_DATASETS_CACHE'] = os.path.join(hf_cache_dir, 'datasets')
    env['HF_HUB_CACHE'] = os.path.join(hf_cache_dir, 'hub')
    
    print(f"Running command: {' '.join(cmd)}")
    if gpu_id is not None:
        print(f"Using GPU {gpu_id}")
    print(f"Hugging Face cache directory: {hf_cache_dir}")
    
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


def setup_logging(log_file=None):
    """Setup logging to both console and file"""
    if log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"experiment_logs/experiments_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger, log_file


def main():
    parser = argparse.ArgumentParser(description='Run Flexible Multi-Task Comorbidity Detection Experiments')
    parser.add_argument('--config_csv', 
                       default='experimentation_plan_simplified.csv',
                       help='Path to experiment configuration CSV')
    parser.add_argument('--biomarker_config',
                       default='config/biomarker_config_default.yaml',
                       help='Path to biomarker configuration file (YAML or JSON)')
    parser.add_argument('--data_dir', 
                       required=True,
                       help='Path to dataset directory containing train.csv and val.csv')
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
    parser.add_argument('--enable_lr_search', 
                       action='store_true',
                       help='Enable learning rate hyperparameter search (expands experiments)')
    parser.add_argument('--learning_rate', 
                       type=float,
                       help='Specific learning rate for single experiment (overrides enable_lr_search)')
    parser.add_argument('--turing1_only', 
                       action='store_true',
                       help='Run only experiments compatible with Turing1 GPUs (RTX 2080 Ti, 11GB)')
    parser.add_argument('--resume', 
                       action='store_true',
                       help='Resume experiments by skipping already completed ones')
    parser.add_argument('--log_file', 
                       help='Log file path (default: experiments_YYYYMMDD_HHMMSS.log)')
    parser.add_argument('--no_confirm', 
                       action='store_true',
                       help='Skip user confirmation (for automated runs)')
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
    
    # Validate arguments
    if args.learning_rate is not None and args.enable_lr_search:
        print("❌ Error: Cannot use both --learning_rate and --enable_lr_search simultaneously")
        print("   Use --learning_rate for a specific learning rate, or --enable_lr_search to test multiple rates")
        sys.exit(1)
    
    if args.learning_rate is not None and not args.model_name:
        print("❌ Error: --learning_rate requires --model_name to be specified")
        print("   Specific learning rates can only be used with specific models")
        sys.exit(1)
    
    # Setup logging
    logger, log_file = setup_logging(args.log_file)
    logger.info(f"🚀 Starting experiment runner - Log file: {log_file}")
    logger.info(f"📋 Arguments: {vars(args)}")
    
    # Check if files exist
    if not os.path.exists(args.config_csv):
        print(f"❌ Configuration file not found: {args.config_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.biomarker_config):
        print(f"❌ Biomarker configuration file not found: {args.biomarker_config}")
        sys.exit(1)
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Load biomarker configuration
    print(f"📊 Loading biomarker configuration from: {args.biomarker_config}")
    try:
        biomarker_config = FlexibleBiomarkerConfig(args.biomarker_config)
        print(f"✅ Biomarker configuration loaded:")
        biomarker_config.print_summary()
    except Exception as e:
        print(f"❌ Error loading biomarker configuration: {e}")
        sys.exit(1)
    
    # Validate dataset compatibility
    print(f"\n🔍 Validating dataset compatibility...")
    try:
        import pandas as pd
        train_df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(args.data_dir, 'val.csv'))
        
        train_compatible, train_missing = biomarker_config.validate_dataset_compatibility(train_df)
        val_compatible, val_missing = biomarker_config.validate_dataset_compatibility(val_df)
        
        if not train_compatible:
            print(f"❌ Train dataset missing columns: {train_missing}")
            sys.exit(1)
        
        if not val_compatible:
            print(f"❌ Validation dataset missing columns: {val_missing}")
            sys.exit(1)
        
        print(f"✅ Dataset compatibility verified")
        print(f"   Train samples: {len(train_df):,}")
        print(f"   Validation samples: {len(val_df):,}")
        print(f"   Required biomarkers: {len(biomarker_config.get_all_biomarker_names())}")
        print(f"   Output tensor size: {biomarker_config.total_output_size}")
        
    except Exception as e:
        print(f"❌ Error validating dataset: {e}")
        sys.exit(1)
    
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
        
        # Handle learning rate specification
        if args.learning_rate is not None:
            # Use specific learning rate (overrides enable_lr_search)
            config.learning_rate = [args.learning_rate]
            # Regenerate experiment name with the specific learning rate
            config.experiment_name = config._generate_experiment_name()
            configs_to_run = [config]
            print(f"🎯 Using specific learning rate: {args.learning_rate} for {args.model_name}")
        elif args.enable_lr_search:
            # Expand learning rates if requested
            configs_to_run = config.generate_lr_experiments()
            print(f"🔍 Learning rate search enabled: {len(configs_to_run)} experiments for {args.model_name}")
        else:
            configs_to_run = [config]
    else:
        # Run all or must-include models
        if args.enable_lr_search:
            if args.must_include_only:
                if args.turing1_only:
                    configs_to_run = config_loader.load_turing1_must_include_with_lr_expansion()
                    print(f"🔍 Learning rate search + Turing1 filter: Expanded to {len(configs_to_run)} experiments")
                else:
                    configs_to_run = config_loader.load_must_include_configs_with_lr_expansion()
                    print(f"🔍 Learning rate search enabled: Expanded to {len(configs_to_run)} experiments")
            else:
                configs_to_run = config_loader.load_all_configs_with_lr_expansion()
                print(f"🔍 Learning rate search enabled: Expanded to {len(configs_to_run)} experiments")
        else:
            if args.must_include_only:
                if args.turing1_only:
                    configs_to_run = config_loader.load_turing1_must_include_configs()
                    print(f"🖥️ Turing1-compatible must-include experiments: {len(configs_to_run)} experiments")
                else:
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
    print("=" * 100)
    print(f"{'#':<3} {'Model':<40} {'Memory':<12} {'LR':<10} {'Status':<15}")
    print("-" * 100)
    for i, config in enumerate(configs_to_run):
        gpu_str = f"GPU {gpu_assignments.get(config.model, 'auto')}" if args.check_memory else "auto"
        output_dir = os.path.join(args.output_base_dir, config.experiment_name)
        exists_str = "EXISTS" if os.path.exists(output_dir) else "NEW"
        lr_str = f"{config.learning_rate[0]:.0e}" if config.learning_rate else "N/A"
        print(f"{i+1:<3} {config.model:<40} {config.expected_gpu_memory:<12} {lr_str:<10} {exists_str:<15}")
    print("=" * 100)
    
    print(f"\nBiomarker Configuration: {biomarker_config.experiment_name}")
    print(f"Binary tasks: {biomarker_config.num_binary_tasks}")
    print(f"Multiclass tasks: {biomarker_config.num_multiclass_tasks}")
    print(f"Continuous tasks: {biomarker_config.num_continuous_tasks}")
    print(f"Total output size: {biomarker_config.total_output_size}")
    
    if args.dry_run:
        print("🔍 Dry run completed. Remove --dry_run to actually run experiments.")
        return
    
    # Filter out completed experiments if resume is enabled
    if args.resume:
        original_count = len(configs_to_run)
        remaining_configs = []
        completed_configs = []
        
        for config in configs_to_run:
            is_completed, marker = is_experiment_completed(config, args.output_base_dir)
            if is_completed:
                completed_configs.append((config.experiment_name, marker))
            else:
                remaining_configs.append(config)
        
        configs_to_run = remaining_configs
        logger.info(f"📋 Resume mode: {len(completed_configs)} already completed, {len(configs_to_run)} remaining")
        
        if completed_configs:
            logger.info("✅ Already completed experiments:")
            for exp_name, marker in completed_configs:
                logger.info(f"  - {exp_name} (found: {os.path.basename(marker)})")
    
    if len(configs_to_run) == 0:
        logger.info("🎉 All experiments already completed!")
        return
    
    # Confirm before running
    if not args.model_name and not args.no_confirm:  # Only ask for confirmation when running multiple experiments
        response = input(f"\n🚀 Ready to run {len(configs_to_run)} experiments. Continue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            logger.info("❌ Aborted by user")
            return
    
    # Run experiments
    logger.info(f"\n🏃 Starting experiments...")
    successful_runs = []
    failed_runs = []
    
    for i, config in enumerate(configs_to_run):
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 Running experiment {i+1}/{len(configs_to_run)}: {config.experiment_name}")
        logger.info(f"Model: {config.model}")
        logger.info(f"Learning Rate: {config.learning_rate[0]:.0e}")
        logger.info(f"{'='*60}")
        
        # Determine GPU to use
        gpu_id = gpu_assignments.get(config.model) if args.check_memory else None
        
        # Add csv_path attribute for the run_single_experiment function
        config.csv_path = args.config_csv
        
        success, output = run_single_experiment(
            config=config,
            data_dir=args.data_dir,
            biomarker_config_path=args.biomarker_config,
            output_base_dir=args.output_base_dir,
            epochs=args.epochs,
            gpu_id=gpu_id
        )
        
        if success:
            successful_runs.append(config.experiment_name)
            logger.info(f"✅ COMPLETED: {config.experiment_name}")
        else:
            failed_runs.append((config.experiment_name, output))
            logger.error(f"❌ FAILED: {config.experiment_name}")
            logger.error(f"Error details: {output}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 FINAL EXPERIMENT SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Successful: {len(successful_runs)}")
    logger.info(f"❌ Failed: {len(failed_runs)}")
    logger.info(f"📁 Results saved in: {args.output_base_dir}")
    logger.info(f"📋 Log file: {log_file}")
    
    if successful_runs:
        logger.info(f"\n✅ Successful experiments:")
        for model in successful_runs:
            logger.info(f"    {model}")
    
    if failed_runs:
        logger.info(f"\n❌ Failed experiments:")
        for model, error in failed_runs:
            logger.info(f"    {model}")
            logger.error(f"    Error: {error[:200]}...")
    
    # Final status
    success_rate = len(successful_runs) / len(configs_to_run) * 100 if configs_to_run else 100
    logger.info(f"\n🎯 Final Status: {len(successful_runs)}/{len(configs_to_run)} experiments completed ({success_rate:.1f}% success rate)")
    logger.info(f"📈 View tensorboard logs: tensorboard --logdir {args.output_base_dir}")
    
    if len(successful_runs) == len(configs_to_run):
        logger.info("🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    elif len(successful_runs) > 0:
        logger.info("⚠️  Some experiments failed - check log for details")
    else:
        logger.error("💥 ALL EXPERIMENTS FAILED - check configuration and logs")
    print(f"🧬 Biomarker config used: {args.biomarker_config}")


if __name__ == "__main__":
    main()

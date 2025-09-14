"""
Flexible Training Pipeline for Multi-Task Comorbidity Detection
Uses the flexible biomarker configuration system for any task structure
"""

import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from tqdm import tqdm
import json
import time
import logging
import sys
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Any

# Import our custom modules
from dataset import ClassifierDataset
from model.model_factory import ModelFactory
from model.flexible_multitask_head import FlexibleMultiTaskLoss, FlexibleMetricsCalculator
from config.biomarker_config import FlexibleBiomarkerConfig
from config.experiment_config import (
    ExperimentConfigLoader, ExperimentConfig, 
    parse_augmentation_string, create_optimizer, create_scheduler
)
from utils.checkpoints import save_checkpoint, load_checkpoint

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def compute_class_weights_for_dataset(dataset, biomarker_config: FlexibleBiomarkerConfig):
    """Compute class weights for balanced training"""
    class_weights = {}
    
    # Get all target tensors
    all_targets = dataset.targets
    
    # Compute weights for binary tasks
    for biomarker in biomarker_config.binary_biomarkers:
        layout = biomarker_config.get_tensor_layout()[biomarker.name]
        labels = all_targets[:, layout.start_idx]
        
        unique_classes = np.unique(labels)
        if len(unique_classes) > 1:
            weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
            # Use positive class weight for BCE
            pos_weight = weights[1] / weights[0] if len(weights) > 1 else 1.0
            class_weights[biomarker.name] = pos_weight
        else:
            class_weights[biomarker.name] = 1.0
    
    return class_weights


def create_data_transforms(config: ExperimentConfig, is_training=True):
    """Create data transforms based on configuration"""
    aug_params = parse_augmentation_string(config.image_augmentations)
    
    if is_training:
        transform_list = []
        
        # Add augmentations
        if aug_params['horizontal_flip']:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        if aug_params['rotation'] > 0:
            transform_list.append(transforms.RandomRotation(degrees=aug_params['rotation']))
        
        if aug_params['random_crop']:
            transform_list.extend([
                transforms.RandomResizedCrop(256, scale=(0.8, 1.0), ratio=(0.75, 1.33))
            ])
        
        if aug_params['color_jitter']:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=aug_params['brightness'],
                    contrast=aug_params['contrast']
                )
            )
        
        # Always add tensor conversion
        transform_list.append(transforms.ToTensor())
        
        # Add normalization
        if aug_params['imagenet_norm']:
            # Use CT-specific normalization
            transform_list.append(transforms.Normalize((0.55001191,), (0.18854326,)))
        
        return transforms.Compose(transform_list)
    
    else:
        # Validation/test transforms (no augmentation)
        transform_list = [transforms.ToTensor()]
        
        if aug_params['imagenet_norm']:
            transform_list.append(transforms.Normalize((0.55001191,), (0.18854326,)))
        
        return transforms.Compose(transform_list)


def setup_logging(output_dir: str, experiment_name: str):
    """Set up comprehensive logging for the experiment"""
    # Create logs directory
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up main logger
    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    # File handler for detailed logs
    detailed_log_file = os.path.join(logs_dir, 'experiment_detailed.log')
    file_handler = logging.FileHandler(detailed_log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Create separate logger for training progress
    training_logger = logging.getLogger('training')
    training_logger.setLevel(logging.INFO)
    training_logger.handlers.clear()
    
    training_log_file = os.path.join(logs_dir, 'training_progress.log')
    training_handler = logging.FileHandler(training_log_file)
    training_handler.setLevel(logging.INFO)
    training_handler.setFormatter(simple_formatter)
    training_logger.addHandler(training_handler)
    training_logger.addHandler(console_handler)
    
    # Log experiment start
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Logs directory: {logs_dir}")
    
    return logger, training_logger


def create_balanced_sampler(dataset, biomarker_config: FlexibleBiomarkerConfig):
    """Create balanced sampler for training"""
    # Get all target tensors
    all_targets = dataset.targets
    
    # Create sample weights based on inverse frequency
    sample_weights = np.ones(len(dataset))
    
    # Weight based on binary biomarkers
    for biomarker in biomarker_config.binary_biomarkers:
        layout = biomarker_config.get_tensor_layout()[biomarker.name]
        labels = all_targets[:, layout.start_idx]
        
        unique, counts = np.unique(labels, return_counts=True)
        class_weights = len(labels) / (len(unique) * counts)
        
        for j, label in enumerate(labels):
            sample_weights[j] *= class_weights[int(label)]
    
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def train_epoch(model, dataloader, criterion, optimizer, device, metrics_calc):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    all_predictions = []
    all_targets = []
    loss_components = {'total_loss': 0}
    
    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        targets = targets.to(device)
        
        # Convert single channel to 3-channel for models expecting RGB
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Forward pass
        predictions = model(images)
        
        # Calculate loss
        loss, loss_dict = criterion(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        for key, value in loss_dict.items():
            if key not in loss_components:
                loss_components[key] = 0
            loss_components[key] += value
        
        # Store predictions and targets for metric calculation
        all_predictions.append(predictions.detach().cpu())
        all_targets.append(targets.detach().cpu())
    
    # Calculate metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = metrics_calc.calculate_all_metrics(all_predictions, all_targets)
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    for key in loss_components:
        loss_components[key] /= len(dataloader)
    
    return avg_loss, metrics, loss_components


def validate_epoch(model, dataloader, criterion, device, metrics_calc):
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0
    all_predictions = []
    all_targets = []
    loss_components = {'total_loss': 0}
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Validation")):
            images = images.to(device)
            targets = targets.to(device)
            
            # Convert single channel to 3-channel for models expecting RGB
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            
            # Forward pass
            predictions = model(images)
            
            # Calculate loss
            loss, loss_dict = criterion(predictions, targets)
            
            # Accumulate metrics
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value
            
            # Store predictions and targets for metric calculation
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())
    
    # Calculate metrics with threshold optimization
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Update optimal thresholds based on validation data
    metrics_calc.update_optimal_thresholds(all_predictions, all_targets)
    
    # Calculate metrics using optimal thresholds
    metrics = metrics_calc.calculate_all_metrics(all_predictions, all_targets)
    
    # Average losses
    avg_loss = total_loss / len(dataloader)
    for key in loss_components:
        loss_components[key] /= len(dataloader)
    
    return avg_loss, metrics, loss_components


def train_model(config: ExperimentConfig, data_dir: str, output_dir: str, 
                biomarker_config: FlexibleBiomarkerConfig, epochs: int = 100):
    """Main training function"""
    
    # Safety check: if directory exists and has important files, create a new one
    if os.path.exists(output_dir):
        important_files = ['best_checkpoint.pth', 'config.json', 'experiment_results.csv']
        has_important_files = any(os.path.exists(os.path.join(output_dir, f)) for f in important_files)
        
        if has_important_files:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            original_output_dir = output_dir
            output_dir = f"{output_dir}_{timestamp}"
            print(f"⚠️  Output directory {original_output_dir} exists with important files.")
            print(f"📁 Using new directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup comprehensive logging
    logger, training_logger = setup_logging(output_dir, config.experiment_name)
    
    # Setup tensorboard logging
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))
    
    # Save configuration
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Configuration saved to: {config_file}")
    
    # Save biomarker configuration
    biomarker_config_file = os.path.join(output_dir, 'biomarker_config.json')
    biomarker_config.save_to_file(biomarker_config_file)
    logger.info(f"Biomarker configuration saved to: {biomarker_config_file}")
    
    logger.info(f"Model: {config.model}")
    logger.info(f"Expected GPU memory: {config.expected_gpu_memory}")
    logger.info(f"Training epochs: {epochs}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Biomarker configuration: {biomarker_config.experiment_name}")
    logger.info(f"Total output size: {biomarker_config.total_output_size}")
    
    # Create data transforms
    train_transform = create_data_transforms(config, is_training=True)
    val_transform = create_data_transforms(config, is_training=False)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = ClassifierDataset(
        data_dir, biomarker_config, transforms=train_transform, 
        size=256, train=True
    )
    
    val_dataset = ClassifierDataset(
        data_dir, biomarker_config, transforms=val_transform,
        size=256, train=False
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Compute class weights if specified
    class_weights = None
    if config.class_weighting == 'inverse_frequency':
        logger.info("Computing class weights...")
        class_weights = compute_class_weights_for_dataset(train_dataset, biomarker_config)
        logger.info(f"Class weights computed for {len(class_weights)} binary biomarkers")
    
    # Create data loaders
    if config.sampling_strategy == 'balanced_batch':
        train_sampler = create_balanced_sampler(train_dataset, biomarker_config)
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, 
            sampler=train_sampler, num_workers=8, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, 
            shuffle=True, num_workers=8, pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, 
        shuffle=False, num_workers=8, pin_memory=True
    )
    
    # Create model
    logger.info("Creating model...")
    model = ModelFactory.create_model(
        architecture=config.model,
        num_classes=biomarker_config.total_output_size,
        pretrained_weights=config.pretrained_weights,
        fine_tuning_strategy=config.fine_tuning_strategy,
        dropout=config.dropout,
        biomarker_config=biomarker_config
    )

    model = model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    criterion = FlexibleMultiTaskLoss(biomarker_config, class_weights=class_weights)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model.parameters(), config)
    scheduler = create_scheduler(optimizer, config, epochs)
    
    # Create metrics calculator
    metrics_calc = FlexibleMetricsCalculator(biomarker_config)
    
    # Training loop
    best_avg_auroc = 0.0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    logger.info(f"Starting training for {epochs} epochs with early stopping (patience: {patience})")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        training_logger.info(f"Starting Epoch {epoch+1}/{epochs}")
        
        # Training phase
        train_loss, train_metrics, train_loss_components = train_epoch(
            model, train_loader, criterion, optimizer, device, metrics_calc
        )
        
        # Validation phase
        val_loss, val_metrics, val_loss_components = validate_epoch(
            model, val_loader, criterion, device, metrics_calc
        )
        
        # Update scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Metrics/Average_AUROC_Train', train_metrics['average_auroc'], epoch)
        writer.add_scalar('Metrics/Average_AUROC_Val', val_metrics['average_auroc'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log individual biomarker metrics
        for biomarker in biomarker_config.binary_biomarkers:
            if biomarker.name in train_metrics and biomarker.name in val_metrics:
                train_biomarker_metrics = train_metrics[biomarker.name]
                val_biomarker_metrics = val_metrics[biomarker.name]
                
                if 'auroc' in train_biomarker_metrics and 'auroc' in val_biomarker_metrics:
                    writer.add_scalar(f'AUROC_Train/{biomarker.name}', train_biomarker_metrics['auroc'], epoch)
                    writer.add_scalar(f'AUROC_Val/{biomarker.name}', val_biomarker_metrics['auroc'], epoch)
        
        # Log epoch results
        training_logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        training_logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        training_logger.info(f"Train Avg AUROC: {train_metrics['average_auroc']:.4f}, Val Avg AUROC: {val_metrics['average_auroc']:.4f}")
        
        # Log individual biomarker validation metrics
        training_logger.info("Validation metrics per biomarker:")
        for biomarker in biomarker_config.binary_biomarkers:
            if biomarker.name in val_metrics:
                biomarker_metrics = val_metrics[biomarker.name]
                if 'auroc' in biomarker_metrics and 'accuracy' in biomarker_metrics:
                    training_logger.info(f"  {biomarker.name}: AUROC={biomarker_metrics['auroc']:.4f}, "
                                       f"Acc={biomarker_metrics['accuracy']:.4f}, "
                                       f"F1={biomarker_metrics.get('f1', 0.0):.4f}")
        
        # Log multiclass biomarker metrics if any exist
        for biomarker in biomarker_config.multiclass_biomarkers:
            if biomarker.name in val_metrics:
                biomarker_metrics = val_metrics[biomarker.name]
                if 'accuracy' in biomarker_metrics:
                    training_logger.info(f"  {biomarker.name}: Acc={biomarker_metrics['accuracy']:.4f}, "
                                       f"F1={biomarker_metrics.get('f1_weighted', 0.0):.4f}")
        
        # Log continuous biomarker metrics if any exist
        for biomarker in biomarker_config.continuous_biomarkers:
            if biomarker.name in val_metrics:
                biomarker_metrics = val_metrics[biomarker.name]
                if 'mse' in biomarker_metrics:
                    training_logger.info(f"  {biomarker.name}: MSE={biomarker_metrics['mse']:.4f}, "
                                       f"MAE={biomarker_metrics.get('mae', 0.0):.4f}")
        
        # Save checkpoint if best model
        is_best = val_metrics['average_auroc'] > best_avg_auroc
        if is_best:
            best_avg_auroc = val_metrics['average_auroc']
            best_epoch = epoch + 1
            patience_counter = 0  # Reset patience counter
            training_logger.info(f"🎯 New best model! Average AUROC: {best_avg_auroc:.4f}")
        else:
            patience_counter += 1
            training_logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
            
            # Early stopping check
            if patience_counter >= patience:
                training_logger.info(f"⏹️ Early stopping triggered!")
                break
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': config.to_dict(),
            'biomarker_config': biomarker_config.experiment_name,
            'best_avg_auroc': best_avg_auroc,
            'best_epoch': best_epoch
        }
        
        # Save latest and best checkpoints
        torch.save(checkpoint, os.path.join(output_dir, 'latest_checkpoint.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(output_dir, 'best_checkpoint.pth'))
    
    logger.info(f"Training completed! Best model at epoch {best_epoch} with AUROC: {best_avg_auroc:.4f}")
    
    # Close tensorboard writer
    writer.close()
    
    return model, best_avg_auroc


def main():
    parser = ArgumentParser(description='Flexible Multi-Task Training')
    parser.add_argument('--config_csv', required=True, help='Path to experiment configuration CSV')
    parser.add_argument('--data_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--biomarker_config', required=True, 
                       help='Path to biomarker configuration file (YAML or JSON)')
    parser.add_argument('--output_base_dir', default='/lfs/turing1/0/mahmedc/Comorbidities-Detection/models', 
                       help='Base directory for model outputs')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--model_name', help='Specific model to train')
    parser.add_argument('--learning_rate', type=float, help='Specific learning rate for this experiment')
    parser.add_argument('--experiment_name', help='Custom experiment name (overrides auto-generated name)')
    
    args = parser.parse_args()
    
    # Load biomarker configuration
    print(f"Loading biomarker configuration from: {args.biomarker_config}")
    biomarker_config = FlexibleBiomarkerConfig(args.biomarker_config)
    
    print(f"Biomarker configuration loaded:")
    biomarker_config.print_summary()
    
    # Load experiment configurations
    config_loader = ExperimentConfigLoader(args.config_csv)
    
    if args.model_name:
        # Train specific model
        config = config_loader.load_config_by_model(args.model_name)
        if config is None:
            print(f"Model {args.model_name} not found in configuration file")
            return
        
        # Override learning rate if specified
        if args.learning_rate:
            config.learning_rate = [args.learning_rate]
        
        configs_to_run = [config]
    else:
        # Train all must-include models
        configs_to_run = config_loader.load_must_include_configs()
    
    print(f"Found {len(configs_to_run)} experiments to run")
    
    # Run experiments
    results = []
    for i, config in enumerate(configs_to_run):
        print(f"\n{'='*50}")
        print(f"Running experiment {i+1}/{len(configs_to_run)}: {config.model}")
        print(f"{'='*50}")
        
        # Override experiment name if provided
        if args.experiment_name:
            config.experiment_name = args.experiment_name
        
        # Create output directory
        output_dir = os.path.join(args.output_base_dir, config.experiment_name)
        
        try:
            model, best_auroc = train_model(
                config=config,
                data_dir=args.data_dir,
                output_dir=output_dir,
                biomarker_config=biomarker_config,
                epochs=args.epochs
            )
            
            results.append({
                'model': config.model,
                'experiment_name': config.experiment_name,
                'best_avg_auroc': best_auroc,
                'status': 'completed'
            })
            
        except Exception as e:
            print(f"Error training {config.model}: {str(e)}")
            results.append({
                'model': config.model,
                'experiment_name': config.experiment_name,
                'best_avg_auroc': 0.0,
                'status': f'failed: {str(e)}'
            })
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_base_dir, 'experiment_results.csv'), index=False)
    
    print(f"\n{'='*50}")
    print("All experiments completed!")
    print(f"{'='*50}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

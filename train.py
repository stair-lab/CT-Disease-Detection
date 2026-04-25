"""
Flexible Training Pipeline for Multi-Task Comorbidity Detection
Uses the flexible biomarker configuration system for any task structure
"""

import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
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
from model.gradnorm_loss import GradNormLoss, GradNormTrainer
from config.biomarker_config import FlexibleBiomarkerConfig
from config.experiment_config import (
    ExperimentConfig, get_model_defaults, DEFAULT_AUGMENTATIONS,
    parse_augmentation_string, create_optimizer, create_scheduler
)
from utils.checkpoints import save_checkpoint, load_checkpoint

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set global random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def normalize_fine_tuning_strategy(strategy: str) -> str:
    """Normalize fine-tuning strategy values to canonical internal names."""
    normalized = strategy.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"linear_probe", "linearprobe"}:
        return "linear_probe"
    return "full"


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
    """Create data transforms optimized for different pre-trained models on medical images"""
    aug_params = parse_augmentation_string(config.image_augmentations)
    
    if is_training:
        transform_list = []
        
        # Different augmentation strategies based on pre-training source
        if config.pretrained_weights == "RadImageNet":
            # RadImageNet-specific augmentations (more conservative for medical domain)
            if aug_params['horizontal_flip']:
                # Very conservative for medical images
                transform_list.append(transforms.RandomHorizontalFlip(p=0.2))
            
            # Use RandomApply to prevent over-augmentation
            geometric_augs = []
            
            if aug_params['rotation'] > 0:
                # Very conservative rotation for medical images
                geometric_augs.append(transforms.RandomRotation(degrees=aug_params['rotation']//3))  # 1/3 the rotation
            
            if aug_params['random_crop']:
                # Minimal cropping to preserve anatomical features
                geometric_augs.append(transforms.RandomResizedCrop(
                    256, 
                    scale=(0.95, 1.0),     # Very conservative cropping
                    ratio=(0.9, 1.1)       # Minimal aspect ratio change
                ))
            
            # Apply geometric augmentations with low probability
            if geometric_augs:
                transform_list.append(transforms.RandomApply(geometric_augs, p=0.4))
            
            # Minimal color augmentations for medical images
            if aug_params['color_jitter']:
                transform_list.append(transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=aug_params['brightness'] * 0.3,  # Very reduced intensity
                        contrast=aug_params['contrast'] * 0.3       # Very reduced intensity
                    )
                ], p=0.3))  # Very low probability
            
        else:
            # ImageNet or non-pretrained augmentations (more aggressive)
            if aug_params['horizontal_flip']:
                # Reduced probability for medical images (anatomical consistency)
                transform_list.append(transforms.RandomHorizontalFlip(p=0.3))
            
            # Use RandomApply to prevent over-augmentation
            geometric_augs = []
            
            if aug_params['rotation'] > 0:
                # More conservative rotation for medical images
                geometric_augs.append(transforms.RandomRotation(degrees=aug_params['rotation']//2))  # Half the rotation
            
            if aug_params['random_crop']:
                # Less aggressive cropping to preserve anatomical features
                geometric_augs.append(transforms.RandomResizedCrop(
                    256, 
                    scale=(0.9, 1.0),      # Less aggressive cropping
                    ratio=(0.8, 1.2)       # Slightly wider aspect ratio
                ))
            
            # Apply geometric augmentations with moderate probability
            if geometric_augs:
                transform_list.append(transforms.RandomApply(geometric_augs, p=0.6))
            
            # Color augmentations (less critical for grayscale, but helps with domain adaptation)
            if aug_params['color_jitter']:
                transform_list.append(transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=aug_params['brightness'] * 0.5,  # Reduced intensity
                        contrast=aug_params['contrast'] * 0.5       # Reduced intensity
                    )
                ], p=0.4))  # Lower probability
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # CRITICAL: Convert grayscale to 3-channel for pre-trained models
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        
        # Add normalization - use appropriate stats based on pre-training
        if aug_params['imagenet_norm']:
            if config.pretrained_weights == "ImageNet":
                # Use ImageNet normalization for ImageNet pre-trained models
                transform_list.append(transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ))
            elif config.pretrained_weights == "RadImageNet":
                # Use RadImageNet normalization (medical imaging specific)
                # Note: These are estimated values - you may need to adjust based on actual RadImageNet stats
                transform_list.append(transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # Using ImageNet stats as fallback
                    std=[0.229, 0.224, 0.225]   # RadImageNet likely uses similar normalization
                ))
            else:
                # Use CT-specific normalization for non-pretrained models
                transform_list.append(transforms.Normalize(
                    mean=[0.55001191, 0.55001191, 0.55001191], 
                    std=[0.18854326, 0.18854326, 0.18854326]
                ))
        
        return transforms.Compose(transform_list)
    
    else:
        # Validation/test transforms (no augmentation)
        transform_list = [transforms.ToTensor()]
        
        # CRITICAL: Convert grayscale to 3-channel for pre-trained models
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        
        if aug_params['imagenet_norm']:
            if config.pretrained_weights == "ImageNet":
                # Use ImageNet normalization for ImageNet pre-trained models
                transform_list.append(transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                ))
            elif config.pretrained_weights == "RadImageNet":
                # Use RadImageNet normalization (medical imaging specific)
                # Note: These are estimated values - you may need to adjust based on actual RadImageNet stats
                transform_list.append(transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # Using ImageNet stats as fallback
                    std=[0.229, 0.224, 0.225]   # RadImageNet likely uses similar normalization
                ))
            else:
                # Use CT-specific normalization for non-pretrained models
                transform_list.append(transforms.Normalize(
                    mean=[0.55001191, 0.55001191, 0.55001191], 
                    std=[0.18854326, 0.18854326, 0.18854326]
                ))
        
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
    sample_weights = np.ones(len(dataset), dtype=np.float64)
    
    # Weight based on binary biomarkers
    for biomarker in biomarker_config.binary_biomarkers:
        layout = biomarker_config.get_tensor_layout()[biomarker.name]
        labels = all_targets[:, layout.start_idx]
        
        unique, counts = np.unique(labels, return_counts=True)
        class_weights = len(labels) / (len(unique) * counts)
        
        for j, label in enumerate(labels):
            sample_weights[j] *= class_weights[int(label)]
    
    # Convert to torch.DoubleTensor to prevent overflow and match WeightedRandomSampler expectations
    sample_weights_tensor = torch.from_numpy(sample_weights).double()
    
    return WeightedRandomSampler(sample_weights_tensor, len(sample_weights_tensor), replacement=True)


def train_epoch(model, dataloader, criterion, optimizer, device, metrics_calc, gradnorm_trainer=None):
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
        
        # Calculate loss - use GradNorm if available
        if gradnorm_trainer is not None:
            loss, loss_dict = gradnorm_trainer.compute_loss(model, predictions, targets)
        else:
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


def train_model(
    config: ExperimentConfig,
    data_dir: str,
    output_dir: str,
    biomarker_config: FlexibleBiomarkerConfig,
    epochs: int = 100,
    seed: int = 42,
):
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
            print(f"Warning: Output directory {original_output_dir} exists with important files.")
            print(f"Using new directory: {output_dir}")
    
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
    logger.info(f"Single-target strategy: {config.single_target_strategy}")
    logger.info(f"Multi-target strategy: {config.multi_target_strategy}")
    logger.info(f"Expected GPU memory: {config.expected_gpu_memory}")
    logger.info(f"Training epochs: {epochs}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Biomarker configuration: {biomarker_config.experiment_name}")
    logger.info(f"Total output size: {biomarker_config.total_output_size}")
    logger.info(f"Seed: {seed}")
    
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
    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(seed)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    if config.sampling_strategy == 'balanced_batch':
        train_sampler = create_balanced_sampler(train_dataset, biomarker_config)
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, 
            sampler=train_sampler, num_workers=8, pin_memory=True,
            worker_init_fn=_seed_worker, generator=data_loader_generator
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, 
            shuffle=True, num_workers=8, pin_memory=True,
            worker_init_fn=_seed_worker, generator=data_loader_generator
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, 
        shuffle=False, num_workers=8, pin_memory=True,
        worker_init_fn=_seed_worker, generator=data_loader_generator
    )
    
    # Create model
    logger.info("Creating model...")
    model = ModelFactory.create_model(
        architecture=config.model,
        num_classes=biomarker_config.total_output_size,
        pretrained_weights=config.pretrained_weights,
        fine_tuning_strategy=config.fine_tuning_strategy,
        dropout=config.dropout,
        biomarker_config=biomarker_config,
        single_target_strategy=config.single_target_strategy
    )

    model = model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function - check if GradNorm is enabled
    use_gradnorm = getattr(config, 'use_gradnorm', False)
    gradnorm_trainer = None
    
    if use_gradnorm:
        logger.info("Using GradNorm for loss balancing")
        gradnorm_alpha = getattr(config, 'gradnorm_alpha', 0.16)
        gradnorm_update_freq = getattr(config, 'gradnorm_update_freq', 10)
        
        gradnorm_loss = GradNormLoss(
            biomarker_config=biomarker_config,
            class_weights=class_weights,
            alpha=gradnorm_alpha,
            update_weights_every=gradnorm_update_freq,
            initial_task_loss_average_window=20,
            normalize_losses=True,
            restoring_force_factor=0.1
        )
        gradnorm_loss = gradnorm_loss.to(device)  # Move to correct device
        gradnorm_trainer = GradNormTrainer(gradnorm_loss)
        criterion = gradnorm_loss  # For validation
    else:
        criterion = FlexibleMultiTaskLoss(biomarker_config, class_weights=class_weights)
    
    # Create optimizer and scheduler
    # Note: GradNorm task weights are updated manually, not through optimizer
    optimizer = create_optimizer(model.parameters(), config)
    
    scheduler = create_scheduler(optimizer, config, epochs)
    
    # Create metrics calculator
    metrics_calc = FlexibleMetricsCalculator(biomarker_config)
    
    # Training loop
    best_median_auroc = 0.0
    best_mae = float('inf')  # Initialize MAE tracking for continuous-only scenarios
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    # Determine task types for logging and selection
    has_classification_tasks = (len(biomarker_config.binary_biomarkers) > 0 or 
                               len(biomarker_config.multiclass_biomarkers) > 0)
    has_continuous_tasks = len(biomarker_config.continuous_biomarkers) > 0
    
    logger.info(f"Starting training for {epochs} epochs with early stopping (patience: {patience})")
    if has_classification_tasks and has_continuous_tasks:
        logger.info("Multi-task training: Classification + Regression")
    elif has_classification_tasks:
        logger.info("Classification training: Using median AUROC for best model selection")
    elif has_continuous_tasks:
        logger.info("Regression training: Using MAE for best model selection")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        training_logger.info(f"Starting Epoch {epoch+1}/{epochs}")
        
        # Training phase
        train_loss, train_metrics, train_loss_components = train_epoch(
            model, train_loader, criterion, optimizer, device, metrics_calc, gradnorm_trainer
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
        writer.add_scalar('Metrics/Median_AUROC_Train', train_metrics['median_auroc'], epoch)
        writer.add_scalar('Metrics/Median_AUROC_Val', val_metrics['median_auroc'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log GradNorm weights if using GradNorm
        if gradnorm_trainer is not None:
            gradnorm_stats = gradnorm_trainer.get_training_stats()
            task_weights = gradnorm_stats['task_weights']
            
            for task_name, weight in task_weights.items():
                writer.add_scalar(f'GradNorm_Weights/{task_name}', weight, epoch)
            
            # Log GradNorm info
            if gradnorm_stats['initial_losses_computed']:
                training_logger.info(f"GradNorm task weights: {task_weights}")
        
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
        training_logger.info(f"Train Median AUROC: {train_metrics['median_auroc']:.4f}, Val Median AUROC: {val_metrics['median_auroc']:.4f}")
        
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
        # Use MAE for continuous-only scenarios, AUROC for classification scenarios
        
        if has_classification_tasks:
            # Use median AUROC for classification scenarios
            # Check if we actually have AUROC values (not just 0.0)
            if val_metrics['median_auroc'] > 0.0:
                is_best = val_metrics['median_auroc'] > best_median_auroc
                if is_best:
                    best_median_auroc = val_metrics['median_auroc']
                    best_epoch = epoch + 1
                    patience_counter = 0  # Reset patience counter
                    training_logger.info(f"New best model! Median AUROC: {best_median_auroc:.4f} (Avg: {val_metrics['average_auroc']:.4f})")
                else:
                    patience_counter += 1
                    training_logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
                    
                    # Early stopping check
                    if patience_counter >= patience:
                        training_logger.info("Early stopping triggered!")
                        break
            else:
                # No actual AUROC values available - fall back to MAE if continuous tasks exist
                if has_continuous_tasks:
                    training_logger.warning("No AUROC values available for classification tasks, falling back to MAE for model selection")
                    # Use MAE logic (will be handled in the elif block below)
                    pass
                else:
                    # No meaningful metrics available
                    is_best = False
                    patience_counter += 1
                    training_logger.warning("No AUROC values available and no continuous tasks - cannot determine best model")
        elif has_continuous_tasks:
            # Use MAE for continuous-only scenarios (lower MAE is better)
            mae_values = []
            for biomarker in biomarker_config.continuous_biomarkers:
                if biomarker.name in val_metrics:
                    mae = val_metrics[biomarker.name].get('mae', float('inf'))
                    mae_values.append(mae)
                    training_logger.info(f"  {biomarker.name}: MAE={mae:.4f}")
            
            if mae_values:
                current_mae = np.mean(mae_values)
                is_best = current_mae < best_mae
                if is_best:
                    best_mae = current_mae
                    best_epoch = epoch + 1
                    patience_counter = 0  # Reset patience counter
                    training_logger.info(f"New best model! Average MAE: {best_mae:.4f}")
                else:
                    patience_counter += 1
                    training_logger.info(f"No improvement. Current MAE: {current_mae:.4f}, Best MAE: {best_mae:.4f}. Patience: {patience_counter}/{patience}")
                    
                    # Early stopping check
                    if patience_counter >= patience:
                        training_logger.info("Early stopping triggered!")
                        break
            else:
                # Fallback if no MAE values available
                is_best = False
                patience_counter += 1
                training_logger.warning("No MAE values available for continuous biomarkers")
        else:
            # No biomarkers configured - should not happen
            is_best = False
            training_logger.error("No biomarkers configured!")
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
            'best_median_auroc': best_median_auroc,
            'best_mae': best_mae,
            'best_epoch': best_epoch,
            'optimal_thresholds': metrics_calc.optimal_thresholds  # Save optimal thresholds
        }
        
        # Save latest and best checkpoints
        torch.save(checkpoint, os.path.join(output_dir, 'latest_checkpoint.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(output_dir, 'best_checkpoint.pth'))
    
    # Close tensorboard writer
    writer.close()
    
    # Final logging based on task type and actual metrics used
    if has_classification_tasks and best_median_auroc > 0.0:
        # Classification tasks with actual AUROC values
        logger.info(f"Training completed! Best model at epoch {best_epoch} with Median AUROC: {best_median_auroc:.4f}")
        return model, best_median_auroc
    elif has_continuous_tasks:
        # Continuous tasks (either only continuous, or classification tasks fell back to MAE)
        logger.info(f"Training completed! Best model at epoch {best_epoch} with MAE: {best_mae:.4f}")
        return model, best_mae
    else:
        # No meaningful metrics available
        logger.info(f"Training completed! Best model at epoch {best_epoch} (no meaningful metrics available)")
        return model, 0.0


def main():
    parser = ArgumentParser(description='Flexible Multi-Task Training')

    # --- Required arguments ---
    parser.add_argument('--model', required=True,
                        help='Model architecture to train (e.g. "ResNet-18", "ViT-Small (DINOv2)"). '
                             'See README for the full list of supported models.')
    parser.add_argument('--data_dir', required=True,
                        help='Path to dataset directory. Must contain train.csv, val.csv, and a data/ subfolder with PNG images.')
    parser.add_argument('--biomarker_config', required=True,
                        help='Path to biomarker configuration file (YAML or JSON). '
                             'See config/biomarker_config_multitask_example.yaml for the full multi-task config used in the paper.')

    # --- Output ---
    parser.add_argument('--output_dir', default='./outputs',
                        help='Directory to save checkpoints, logs, and TensorBoard (default: ./outputs)')
    parser.add_argument('--experiment_name',
                        help='Custom name for this run. Auto-generated from model/lr/batch if not provided.')

    # --- Training hyperparameters (defaults match the published experiments) ---
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--optimizer', default='AdamW', choices=['AdamW', 'Adam', 'SGD'],
                        help='Optimizer (default: AdamW)')
    parser.add_argument('--scheduler', default='CosineAnnealing',
                        choices=['CosineAnnealing', 'CosineAnnealingWarmRestarts',
                                 'ReduceLROnPlateau', 'StepLR', 'ExponentialLR'],
                        help='LR scheduler (default: CosineAnnealing)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--class_weighting', default='inverse_frequency',
                        choices=['inverse_frequency', 'none'],
                        help='Class weighting strategy (default: inverse_frequency)')
    parser.add_argument('--sampling_strategy', default='balanced_batch',
                        choices=['balanced_batch', 'random'],
                        help='Sampling strategy for training DataLoader (default: balanced_batch)')
    parser.add_argument('--fine_tuning_strategy', default='Full fine-tuning',
                        choices=['Full fine-tuning', 'full', 'linear_probe'],
                        help='Fine-tuning strategy (default: Full fine-tuning)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Global random seed for reproducibility (default: 42)')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic backend behavior (can reduce throughput)')

    # --- Per-model defaults (auto-detected from model name if not specified) ---
    parser.add_argument('--pretrained_weights',
                        help='Pretrained weights to use. Auto-detected from --model if not provided.')
    parser.add_argument('--single_target_strategy',
                        help='Single-target strategy. Auto-detected from --model if not provided.')

    # --- GradNorm ---
    parser.add_argument('--use_gradnorm', action='store_true',
                        help='Enable GradNorm adaptive loss balancing')
    parser.add_argument('--gradnorm_alpha', type=float, default=0.16,
                        help='GradNorm restoring force strength (default: 0.16)')
    parser.add_argument('--gradnorm_update_freq', type=int, default=10,
                        help='Update GradNorm weights every N iterations (default: 10)')

    args = parser.parse_args()
    set_global_seed(args.seed, deterministic=args.deterministic)

    # Load biomarker configuration
    print(f"Loading biomarker configuration from: {args.biomarker_config}")
    biomarker_config = FlexibleBiomarkerConfig(args.biomarker_config)

    print("Biomarker configuration loaded:")
    biomarker_config.print_summary()

    # Resolve per-model defaults for pretrained_weights and single_target_strategy
    model_defaults = get_model_defaults(args.model)
    pretrained_weights = args.pretrained_weights or model_defaults['pretrained_weights']
    single_target_strategy = args.single_target_strategy or model_defaults['single_target_strategy']
    fine_tuning_strategy = normalize_fine_tuning_strategy(args.fine_tuning_strategy)

    # Build ExperimentConfig directly from CLI args
    config = ExperimentConfig(
        model=args.model,
        loss_function='CE',
        must_include=True,
        learning_rate=[args.learning_rate],
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        image_augmentations=DEFAULT_AUGMENTATIONS.copy(),
        dropout=args.dropout,
        loss_specific_params='class_weights=inverse_frequency',
        multi_target_strategy='Shared backbone + task-specific heads',
        single_target_strategy=single_target_strategy,
        pretrained_weights=pretrained_weights,
        fine_tuning_strategy=fine_tuning_strategy,
        expected_gpu_memory='',
        architectural_family='',
        class_weighting=args.class_weighting,
        sampling_strategy=args.sampling_strategy,
        threshold_selection='F1_optimal',
        experiment_name=args.experiment_name or '',
        use_gradnorm=args.use_gradnorm,
        gradnorm_alpha=args.gradnorm_alpha,
        gradnorm_update_freq=args.gradnorm_update_freq,
    )

    output_dir = os.path.join(args.output_dir, config.experiment_name)

    print(f"\n{'='*50}")
    print(f"Training: {config.model}")
    print(f"  Pretrained weights:      {pretrained_weights}")
    print(f"  Single-target strategy:  {single_target_strategy}")
    print(f"  Learning rate:           {args.learning_rate}")
    print(f"  Batch size:              {args.batch_size}")
    print(f"  Epochs:                  {args.epochs}")
    print(f"  Seed:                    {args.seed}")
    print(f"  Deterministic:           {args.deterministic}")
    print(f"  Output dir:              {output_dir}")
    print(f"{'='*50}\n")

    train_model(
        config=config,
        data_dir=args.data_dir,
        output_dir=output_dir,
        biomarker_config=biomarker_config,
        epochs=args.epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

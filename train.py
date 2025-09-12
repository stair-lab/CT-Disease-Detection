"""
Enhanced Training Pipeline for Multi-Task Comorbidity Detection
Includes tensorboard logging, validation passes, and comprehensive checkpointing
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
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
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
from config.experiment_config import (
    ExperimentConfigLoader, ExperimentConfig, 
    parse_augmentation_string, create_optimizer, create_scheduler
)
from config.biomarker_config import BiomarkerConfig, get_default_biomarker_config
from utils.checkpoints import save_checkpoint, load_checkpoint

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define conditions (biomarkers)
CONDITIONS = ['GENDER', 'HCC18', 'HCC22', 'HCC85', 'HCC96', 'HCC108', 'HCC111', 
             'CalciumScoring_AbdominalAgatston', 'AGE', 'RAF']
BINARY_CONDITIONS = CONDITIONS[1:8]  # HCC conditions
CALCIUM_CLASSES = 4
NUM_BINARY_TASKS = 7
NUM_REGRESSION_TASKS = 2


class MultiTaskLoss(nn.Module):
    """Multi-task loss function for comorbidity detection"""
    
    def __init__(self, class_weights=None, calcium_classes=4):
        super().__init__()
        self.class_weights = class_weights
        self.calcium_classes = calcium_classes
        
        # Create weighted BCE loss if class weights provided
        if class_weights is not None:
            # Convert to tensor and move to device
            if isinstance(class_weights, dict):
                # Assume class_weights is per binary task
                self.binary_weights = torch.tensor([class_weights.get(i, 1.0) for i in range(NUM_BINARY_TASKS)], 
                                                 dtype=torch.float32).to(device)
            else:
                self.binary_weights = torch.tensor(class_weights[:NUM_BINARY_TASKS], 
                                                 dtype=torch.float32).to(device)
        else:
            self.binary_weights = None
    
    def forward(self, predictions, targets):
        """
        Calculate multi-task loss
        
        Args:
            predictions: [batch_size, 13] - 7 binary + 4 calcium + 2 regression
            targets: [batch_size, 13] - same format
        """
        batch_size = predictions.size(0)
        
        # Split predictions and targets
        binary_pred = predictions[:, :NUM_BINARY_TASKS]  # [B, 7]
        calcium_pred = predictions[:, NUM_BINARY_TASKS:NUM_BINARY_TASKS+CALCIUM_CLASSES]  # [B, 4]
        regression_pred = predictions[:, -NUM_REGRESSION_TASKS:]  # [B, 2]
        
        binary_target = targets[:, :NUM_BINARY_TASKS]  # [B, 7]
        calcium_target = targets[:, NUM_BINARY_TASKS:NUM_BINARY_TASKS+CALCIUM_CLASSES]  # [B, 4]
        regression_target = targets[:, -NUM_REGRESSION_TASKS:]  # [B, 2]
        
        # Binary classification loss (BCE with logits)
        if self.binary_weights is not None:
            # Apply class weights
            binary_loss = 0
            for i in range(NUM_BINARY_TASKS):
                weight = self.binary_weights[i]
                loss_i = F.binary_cross_entropy_with_logits(
                    binary_pred[:, i], binary_target[:, i], 
                    pos_weight=weight.unsqueeze(0).expand(batch_size)
                )
                binary_loss += loss_i
            binary_loss /= NUM_BINARY_TASKS
        else:
            binary_loss = F.binary_cross_entropy_with_logits(binary_pred, binary_target)
        
        # Multiclass classification loss (Cross-entropy)
        calcium_target_idx = torch.argmax(calcium_target, dim=1)
        calcium_loss = F.cross_entropy(calcium_pred, calcium_target_idx)
        
        # Regression loss (MSE)
        regression_loss = F.mse_loss(regression_pred, regression_target)
        
        # Combine losses with equal weighting (can be made configurable)
        total_loss = binary_loss + calcium_loss + regression_loss
        
        return total_loss, {
            'binary_loss': binary_loss.item(),
            'calcium_loss': calcium_loss.item(),
            'regression_loss': regression_loss.item(),
            'total_loss': total_loss.item()
        }


class MetricsCalculator:
    """Calculate comprehensive metrics for multi-task learning"""
    
    def __init__(self, biomarker_config: BiomarkerConfig):
        self.biomarker_config = biomarker_config
        self.tensor_layout = biomarker_config.get_tensor_layout()
        self.binary_biomarkers = [b.name for b in biomarker_config.binary_biomarkers]
        self.multiclass_biomarkers = [b.name for b in biomarker_config.multiclass_biomarkers]
        self.continuous_biomarkers = [b.name for b in biomarker_config.continuous_biomarkers]
        
    def calculate_binary_metrics(self, predictions, targets, threshold=0.5):
        """Calculate metrics for binary classification tasks"""
        metrics = {}
        
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        for biomarker_name in self.binary_biomarkers:
            layout = self.tensor_layout[biomarker_name]
            idx = layout['start_idx']
            
            pred_probs = predictions[:, idx]
            true_labels = targets[:, idx].astype(int)
            
            # AUROC
            try:
                auroc = roc_auc_score(true_labels, pred_probs)
            except ValueError:
                auroc = 0.0
            
            # Predictions with threshold
            pred_labels = (pred_probs > threshold).astype(int)
            
            # Accuracy
            accuracy = (pred_labels == true_labels).mean()
            
            # Calculate confusion matrix components
            true_positives = np.sum((pred_labels == 1) & (true_labels == 1))
            true_negatives = np.sum((pred_labels == 0) & (true_labels == 0))
            false_positives = np.sum((pred_labels == 1) & (true_labels == 0))
            false_negatives = np.sum((pred_labels == 0) & (true_labels == 1))
            
            # Sensitivity (Recall/True Positive Rate)
            sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            
            # Specificity (True Negative Rate)
            specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
            
            # F1 Score
            try:
                f1 = f1_score(true_labels, pred_labels, zero_division=0.0)
            except (ValueError, ZeroDivisionError):
                f1 = 0.0
            
            # Find optimal F1 threshold
            try:
                precision, recall, thresholds = precision_recall_curve(true_labels, pred_probs)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                optimal_idx = np.argmax(f1_scores)
                optimal_f1 = f1_scores[optimal_idx]
                optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else threshold
            except (ValueError, IndexError):
                optimal_f1 = f1
                optimal_threshold = threshold
            
            metrics[biomarker_name] = {
                'auroc': auroc,
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'f1': f1,
                'optimal_f1': optimal_f1,
                'optimal_threshold': optimal_threshold,
                'true_positives': int(true_positives),
                'true_negatives': int(true_negatives),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives)
            }
        
        return metrics
    
    def calculate_calcium_metrics(self, predictions, targets):
        """Calculate metrics for calcium scoring (multiclass)"""
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Get predicted classes
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(targets, axis=1)
        
        # Overall accuracy
        accuracy = (pred_classes == true_classes).mean()
        
        # Multi-class AUROC
        try:
            auroc = roc_auc_score(targets, predictions, multi_class='ovr', average='macro')
        except ValueError:
            auroc = 0.0
        
        # Per-class sensitivity and specificity
        num_classes = predictions.shape[1]
        class_names = ['ABSENT', 'LOW', 'MEDIUM', 'HIGH']
        per_class_metrics = {}
        
        for class_idx in range(num_classes):
            # Convert to binary classification for this class vs all others
            true_binary = (true_classes == class_idx).astype(int)
            pred_binary = (pred_classes == class_idx).astype(int)
            
            # Calculate confusion matrix components
            true_positives = np.sum((pred_binary == 1) & (true_binary == 1))
            true_negatives = np.sum((pred_binary == 0) & (true_binary == 0))
            false_positives = np.sum((pred_binary == 1) & (true_binary == 0))
            false_negatives = np.sum((pred_binary == 0) & (true_binary == 1))
            
            # Sensitivity and Specificity
            sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
            
            per_class_metrics[f'calcium_{class_names[class_idx].lower()}'] = {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'true_positives': int(true_positives),
                'true_negatives': int(true_negatives),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives)
            }
        
        return {
            'CalciumScoring_AbdominalAgatston': {
                'accuracy': accuracy,
                'auroc': auroc,
                'per_class': per_class_metrics
            }
        }
    
    def calculate_regression_metrics(self, predictions, targets, age_norm=101.0, raf_norm=50.0):
        """Calculate metrics for regression tasks"""
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Denormalize predictions and targets
        age_pred = predictions[:, 0] * age_norm
        age_true = targets[:, 0] * age_norm
        raf_pred = predictions[:, 1] * raf_norm
        raf_true = targets[:, 1] * raf_norm
        
        # Calculate MSE and MAE
        age_mse = np.mean((age_pred - age_true) ** 2)
        age_mae = np.mean(np.abs(age_pred - age_true))
        raf_mse = np.mean((raf_pred - raf_true) ** 2)
        raf_mae = np.mean(np.abs(raf_pred - raf_true))
        
        return {
            'AGE': {
                'mse': age_mse,
                'mae': age_mae
            },
            'RAF': {
                'mse': raf_mse,
                'mae': raf_mae
            }
        }
    
    def calculate_all_metrics(self, predictions, targets):
        """Calculate all metrics"""
        all_metrics = {}
        
        # Calculate binary metrics
        if self.binary_biomarkers:
            binary_metrics = self.calculate_binary_metrics(predictions, targets)
            all_metrics.update(binary_metrics)
        
        # Calculate multiclass metrics
        for biomarker_name in self.multiclass_biomarkers:
            layout = self.tensor_layout[biomarker_name]
            start_idx = layout['start_idx']
            end_idx = layout['end_idx']
            
            multiclass_pred = predictions[:, start_idx:end_idx]
            multiclass_target = targets[:, start_idx:end_idx]
            
            multiclass_metrics = self.calculate_calcium_metrics(multiclass_pred, multiclass_target)
            # Rename the key to the actual biomarker name
            if 'CalciumScoring_AbdominalAgatston' in multiclass_metrics:
                all_metrics[biomarker_name] = multiclass_metrics['CalciumScoring_AbdominalAgatston']
        
        # Calculate regression metrics  
        if self.continuous_biomarkers:
            regression_metrics = self.calculate_regression_metrics(predictions, targets)
            all_metrics.update(regression_metrics)
        
        # Calculate average AUROC for model selection
        auroc_values = []
        
        # Collect AUROC from binary biomarkers
        for biomarker_name in self.binary_biomarkers:
            if biomarker_name in all_metrics and 'auroc' in all_metrics[biomarker_name]:
                auroc_values.append(all_metrics[biomarker_name]['auroc'])
        
        # Collect AUROC from multiclass biomarkers
        for biomarker_name in self.multiclass_biomarkers:
            if biomarker_name in all_metrics and 'auroc' in all_metrics[biomarker_name]:
                auroc_values.append(all_metrics[biomarker_name]['auroc'])
        
        avg_auroc = np.mean(auroc_values) if auroc_values else 0.0
        all_metrics['average_auroc'] = avg_auroc
        
        return all_metrics


def compute_class_weights_for_dataset(dataset, conditions):
    """Compute class weights for balanced training"""
    class_weights = {}
    
    # Get all labels
    all_labels = []
    for i in range(len(dataset)):
        _, labels = dataset[i]
        all_labels.append(labels.numpy())
    
    all_labels = np.array(all_labels)
    
    # Compute weights for binary tasks
    for i, condition in enumerate(conditions[:NUM_BINARY_TASKS]):
        labels = all_labels[:, i]
        unique_classes = np.unique(labels)
        if len(unique_classes) > 1:
            weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
            # Use positive class weight for BCE
            pos_weight = weights[1] / weights[0] if len(weights) > 1 else 1.0
            class_weights[i] = pos_weight
        else:
            class_weights[i] = 1.0
    
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
            # Use CT-specific normalization (from original code)
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
    
    # File handler for training progress
    training_log_file = os.path.join(logs_dir, 'training_progress.log')
    training_handler = logging.FileHandler(training_log_file)
    training_handler.setLevel(logging.INFO)
    training_handler.setFormatter(simple_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Create separate logger for training progress
    training_logger = logging.getLogger('training')
    training_logger.setLevel(logging.INFO)
    training_logger.handlers.clear()
    training_logger.addHandler(training_handler)
    training_logger.addHandler(console_handler)
    
    # Log experiment start
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Logs directory: {logs_dir}")
    
    return logger, training_logger


def create_balanced_sampler(dataset, conditions):
    """Create balanced sampler for training"""
    # Get all labels for binary tasks
    all_labels = []
    for i in range(len(dataset)):
        _, labels = dataset[i]
        # Use only binary labels for balancing
        binary_labels = labels[:NUM_BINARY_TASKS].numpy()
        all_labels.append(binary_labels)
    
    all_labels = np.array(all_labels)
    
    # Create sample weights based on inverse frequency
    sample_weights = np.ones(len(dataset))
    
    for i in range(NUM_BINARY_TASKS):
        labels = all_labels[:, i]
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
    loss_components = {'binary_loss': 0, 'calcium_loss': 0, 'regression_loss': 0, 'total_loss': 0}
    
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
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        
        # Store predictions and targets for metric calculation
        all_predictions.append(torch.sigmoid(predictions).detach().cpu())
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
    loss_components = {'binary_loss': 0, 'calcium_loss': 0, 'regression_loss': 0, 'total_loss': 0}
    
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
            for key in loss_components:
                loss_components[key] += loss_dict[key]
            
            # Store predictions and targets for metric calculation
            all_predictions.append(torch.sigmoid(predictions).detach().cpu())
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


def train_model(config: ExperimentConfig, data_dir: str, output_dir: str, 
                biomarker_config: BiomarkerConfig, epochs: int = 100):
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
    biomarker_config.save_to_json(biomarker_config_file)
    logger.info(f"Biomarker configuration saved to: {biomarker_config_file}")
    
    logger.info(f"Model: {config.model}")
    logger.info(f"Expected GPU memory: {config.expected_gpu_memory}")
    logger.info(f"Training epochs: {epochs}")
    logger.info(f"Data directory: {data_dir}")
    
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
        class_weights = compute_class_weights_for_dataset(train_dataset, CONDITIONS)
        logger.info(f"Class weights: {class_weights}")
    
    # Create data loaders
    if config.sampling_strategy == 'balanced_batch':
        train_sampler = create_balanced_sampler(train_dataset, CONDITIONS)
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
        dropout=config.dropout
    )

    model = model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model architecture: {config.model}")
    logger.info(f"Pretrained weights: {config.pretrained_weights}")
    logger.info(f"Fine-tuning strategy: {config.fine_tuning_strategy}")
    logger.info(f"Dropout: {config.dropout}")
    
    # Create loss function
    criterion = MultiTaskLoss(class_weights=class_weights, calcium_classes=CALCIUM_CLASSES)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model.parameters(), config)
    scheduler = create_scheduler(optimizer, config, epochs)
    
    # Create metrics calculator
    metrics_calc = MetricsCalculator(biomarker_config)
    
    # Training loop
    best_avg_auroc = 0.0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    logger.info(f"Starting training for {epochs} epochs with early stopping (patience: {patience})")
    logger.info(f"Optimizer: {config.optimizer}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Scheduler: {config.scheduler}")
    
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
        writer.add_scalar('Loss/Train_Binary', train_loss_components['binary_loss'], epoch)
        writer.add_scalar('Loss/Train_Calcium', train_loss_components['calcium_loss'], epoch)
        writer.add_scalar('Loss/Train_Regression', train_loss_components['regression_loss'], epoch)
        writer.add_scalar('Loss/Val_Binary', val_loss_components['binary_loss'], epoch)
        writer.add_scalar('Loss/Val_Calcium', val_loss_components['calcium_loss'], epoch)
        writer.add_scalar('Loss/Val_Regression', val_loss_components['regression_loss'], epoch)
        writer.add_scalar('Metrics/Average_AUROC_Train', train_metrics['average_auroc'], epoch)
        writer.add_scalar('Metrics/Average_AUROC_Val', val_metrics['average_auroc'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log individual biomarker metrics
        for condition in BINARY_CONDITIONS:
            if condition in train_metrics and condition in val_metrics:
                if 'auroc' in train_metrics[condition] and 'auroc' in val_metrics[condition]:
                    writer.add_scalar(f'AUROC_Train/{condition}', train_metrics[condition]['auroc'], epoch)
                    writer.add_scalar(f'AUROC_Val/{condition}', val_metrics[condition]['auroc'], epoch)
                if 'f1' in train_metrics[condition] and 'f1' in val_metrics[condition]:
                    writer.add_scalar(f'F1_Train/{condition}', train_metrics[condition]['f1'], epoch)
                    writer.add_scalar(f'F1_Val/{condition}', val_metrics[condition]['f1'], epoch)
                if 'accuracy' in train_metrics[condition] and 'accuracy' in val_metrics[condition]:
                    writer.add_scalar(f'Accuracy_Train/{condition}', train_metrics[condition]['accuracy'], epoch)
                    writer.add_scalar(f'Accuracy_Val/{condition}', val_metrics[condition]['accuracy'], epoch)
                if 'sensitivity' in train_metrics[condition] and 'sensitivity' in val_metrics[condition]:
                    writer.add_scalar(f'Sensitivity_Train/{condition}', train_metrics[condition]['sensitivity'], epoch)
                    writer.add_scalar(f'Sensitivity_Val/{condition}', val_metrics[condition]['sensitivity'], epoch)
                if 'specificity' in train_metrics[condition] and 'specificity' in val_metrics[condition]:
                    writer.add_scalar(f'Specificity_Train/{condition}', train_metrics[condition]['specificity'], epoch)
                    writer.add_scalar(f'Specificity_Val/{condition}', val_metrics[condition]['specificity'], epoch)
        
        # Log calcium scoring metrics
        if 'CalciumScoring_AbdominalAgatston' in val_metrics:
            calcium_metrics = val_metrics['CalciumScoring_AbdominalAgatston']
            if 'accuracy' in calcium_metrics:
                writer.add_scalar('Accuracy_Val/CalciumScoring', calcium_metrics['accuracy'], epoch)
            if 'auroc' in calcium_metrics:
                writer.add_scalar('AUROC_Val/CalciumScoring', calcium_metrics['auroc'], epoch)
        
        # Log epoch results
        training_logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        training_logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        training_logger.info(f"Train Avg AUROC: {train_metrics['average_auroc']:.4f}, Val Avg AUROC: {val_metrics['average_auroc']:.4f}")
        training_logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Log detailed validation metrics per biomarker
        training_logger.info("Validation Metrics per Biomarker:")
        training_logger.info("=" * 80)
        training_logger.info(f"{'Biomarker':<15} {'AUROC':<8} {'Accuracy':<8} {'Sensitivity':<12} {'Specificity':<12}")
        training_logger.info("-" * 80)
        
        for biomarker_name in metrics_calc.binary_biomarkers:
            if biomarker_name in val_metrics:
                metrics = val_metrics[biomarker_name]
                auroc = metrics.get('auroc', 0.0)
                accuracy = metrics.get('accuracy', 0.0)
                sensitivity = metrics.get('sensitivity', 0.0)
                specificity = metrics.get('specificity', 0.0)
                
                training_logger.info(f"{biomarker_name:<15} {auroc:<8.4f} {accuracy:<8.4f} {sensitivity:<12.4f} {specificity:<12.4f}")
            else:
                training_logger.info(f"{biomarker_name:<15} {'N/A':<8} {'N/A':<8} {'N/A':<12} {'N/A':<12}")
        
        # Log multiclass biomarker metrics
        for biomarker_name in metrics_calc.multiclass_biomarkers:
            if biomarker_name in val_metrics:
                multiclass_metrics = val_metrics[biomarker_name]
                auroc = multiclass_metrics.get('auroc', 0.0)
                accuracy = multiclass_metrics.get('accuracy', 0.0)
                display_name = biomarker_name.replace('_', ' ')[:14]  # Truncate for display
                training_logger.info(f"{display_name:<15} {auroc:<8.4f} {accuracy:<8.4f} {'N/A':<12} {'N/A':<12}")
                
                # Log per-class metrics if available
                if 'per_class' in multiclass_metrics:
                    training_logger.info(f"{biomarker_name} Per-Class Metrics:")
                    training_logger.info(f"{'Class':<10} {'Sensitivity':<12} {'Specificity':<12}")
                    training_logger.info("-" * 35)
                    for class_name, class_metrics in multiclass_metrics['per_class'].items():
                        sens = class_metrics.get('sensitivity', 0.0)
                        spec = class_metrics.get('specificity', 0.0)
                        display_name = class_name.replace('calcium_', '').upper()
                        training_logger.info(f"{display_name:<10} {sens:<12.4f} {spec:<12.4f}")
        
        training_logger.info("=" * 80)
        
        # Save checkpoint if best model
        is_best = val_metrics['average_auroc'] > best_avg_auroc
        if is_best:
            best_avg_auroc = val_metrics['average_auroc']
            best_epoch = epoch + 1
            patience_counter = 0  # Reset patience counter
            training_logger.info(f"🎯 New best model! Average AUROC: {best_avg_auroc:.4f}")
            logger.info(f"New best model saved at epoch {best_epoch} with average AUROC: {best_avg_auroc:.4f}")
        else:
            patience_counter += 1
            training_logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
            
            # Early stopping check
            if patience_counter >= patience:
                training_logger.info(f"⏹️ Early stopping triggered! No improvement for {patience} epochs.")
                training_logger.info(f"Best model was at epoch {best_epoch} with average AUROC: {best_avg_auroc:.4f}")
                logger.info(f"Training stopped early after {epoch + 1} epochs due to no improvement")
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
            'best_avg_auroc': best_avg_auroc,
            'best_epoch': best_epoch,
            'patience_counter': patience_counter,
            'patience': patience
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(output_dir, 'latest_checkpoint.pth'))
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(output_dir, 'best_checkpoint.pth'))
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Training completion message and final logging
    if patience_counter >= patience:
        logger.info(f"Training stopped early after {epoch + 1} epochs due to no improvement.")
        training_logger.info(f"🏁 Training stopped early after {epoch + 1} epochs")
    else:
        logger.info(f"Training completed after {epochs} epochs!")
        training_logger.info(f"🏁 Training completed after {epochs} epochs!")
    
    logger.info(f"Best model at epoch {best_epoch} with average AUROC: {best_avg_auroc:.4f}")
    training_logger.info(f"🏆 Best model at epoch {best_epoch} with average AUROC: {best_avg_auroc:.4f}")
    
    # Save comprehensive training summary
    summary_file = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Experiment Name: {config.experiment_name}\n")
        f.write(f"Model: {config.model}\n")
        f.write(f"Total epochs trained: {epoch + 1}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best average AUROC: {best_avg_auroc:.4f}\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Expected GPU memory: {config.expected_gpu_memory}\n")
        f.write(f"Pretrained weights: {config.pretrained_weights}\n")
        f.write(f"Fine-tuning strategy: {config.fine_tuning_strategy}\n")
        f.write(f"Learning rate: {config.learning_rate}\n")
        f.write(f"Batch size: {config.batch_size}\n")
        f.write(f"Optimizer: {config.optimizer}\n")
        f.write(f"Scheduler: {config.scheduler}\n")
        f.write(f"Dropout: {config.dropout}\n")
        f.write(f"Class weighting: {config.class_weighting}\n")
        f.write(f"Sampling strategy: {config.sampling_strategy}\n")
        f.write(f"Train dataset size: {len(train_dataset)}\n")
        f.write(f"Validation dataset size: {len(val_dataset)}\n")
        f.write("\n" + "=" * 80 + "\n")
    
    # Log final model performance summary
    logger.info("=== EXPERIMENT SUMMARY ===")
    logger.info(f"Model: {config.model}")
    logger.info(f"Total epochs trained: {epoch + 1}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best average AUROC: {best_avg_auroc:.4f}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Training summary saved to: {summary_file}")
    logger.info("=== END EXPERIMENT ===")
    
    # Close tensorboard writer
    writer.close()
    
    # Close logging handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    for handler in training_logger.handlers[:]:
        handler.close()
        training_logger.removeHandler(handler)
    
    return model, best_avg_auroc


def main():
    parser = ArgumentParser(description='Enhanced Multi-Task Training')
    parser.add_argument('--config_csv', required=True, help='Path to experiment configuration CSV')
    parser.add_argument('--data_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--output_base_dir', default='/lfs/turing1/0/mahmedc/Comorbidities-Detection/models', 
                       help='Base directory for model outputs')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--model_name', help='Specific model to train (if not provided, trains all must-include models)')
    parser.add_argument('--biomarker_config', default='config/biomarker_config_default.yaml',
                       help='Path to biomarker configuration file (YAML or JSON)')
    
    args = parser.parse_args()
    
    # Load biomarker configuration
    print(f"Loading biomarker configuration from: {args.biomarker_config}")
    if args.biomarker_config.endswith('.yaml') or args.biomarker_config.endswith('.yml'):
        biomarker_config = BiomarkerConfig.load_from_yaml(args.biomarker_config)
    elif args.biomarker_config.endswith('.json'):
        biomarker_config = BiomarkerConfig.load_from_json(args.biomarker_config)
    else:
        print("Warning: Biomarker config file extension not recognized. Using default configuration.")
        biomarker_config = get_default_biomarker_config()
    
    print(f"Biomarker configuration loaded:")
    print(f"  Binary biomarkers: {[b.name for b in biomarker_config.binary_biomarkers]}")
    print(f"  Multiclass biomarkers: {[b.name for b in biomarker_config.multiclass_biomarkers]}")  
    print(f"  Continuous biomarkers: {[b.name for b in biomarker_config.continuous_biomarkers]}")
    print(f"  Total output size: {biomarker_config.total_output_size}")
    
    # Load experiment configurations
    config_loader = ExperimentConfigLoader(args.config_csv)
    
    if args.model_name:
        # Train specific model
        config = config_loader.load_config_by_model(args.model_name)
        if config is None:
            print(f"Model {args.model_name} not found in configuration file")
            return
        configs_to_run = [config]
    else:
        # Train all must-include models
        configs_to_run = config_loader.load_must_include_configs()
    
    print(f"Found {len(configs_to_run)} experiments to run")
    
    # Run experiments
    results = []
    for i, config in enumerate(configs_to_run):
        print(f"\n{'='*50}")
        print(f"Running experiment {i+1}/{len(configs_to_run)}")
        print(f"{'='*50}")
        
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

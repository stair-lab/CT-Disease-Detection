"""
Flexible Multi-Task Testing Script
Supports any biomarker configuration and model architecture
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import json
from typing import Dict, Any, List, Tuple

from dataset import ClassifierDataset, PredictionDataset
from model.model_factory import ModelFactory
from model.flexible_multitask_head import FlexibleMetricsCalculator
from config.biomarker_config import FlexibleBiomarkerConfig
from config.experiment_config import ExperimentConfig, DEFAULT_AUGMENTATIONS
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

try:
    from safetensors.torch import load_file as safetensors_load_file
except ImportError:
    safetensors_load_file = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def arg_parse():
    parser = ArgumentParser(description='Flexible Multi-Task Testing')
    parser.add_argument('--data_dir', required=True, help='Directory with test data')
    parser.add_argument(
        '--checkpoint_path',
        required=True,
        help='Path to model checkpoint (.pth/.pt or .safetensors).'
    )
    parser.add_argument('--biomarker_config', required=True, help='Path to biomarker configuration file (YAML or JSON)')
    parser.add_argument('--output_dir', default='test_results', help='Output directory for results')
    parser.add_argument('--size', default=256, type=int, help='Image size')
    parser.add_argument('--only_pred', action='store_true', help='Only generate predictions (no ground truth evaluation)')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for inference')
    parser.add_argument('--save_predictions', action='store_true', help='Save individual predictions to CSV')
    parser.add_argument('--save_metrics', action='store_true', help='Save detailed metrics to JSON file')
    parser.add_argument('--use_val_for_thresholds', action='store_true', 
                       help='Use validation set for threshold optimization (default: use same data_dir)')
    parser.add_argument('--val_data_dir', help='Path to validation data directory (if different from data_dir)')
    parser.add_argument('--test_csv', default='test.csv', help='CSV file to use for testing (default: test.csv)')
    parser.add_argument(
        '--legacy_checkpoint_compat',
        action='store_true',
        help='Enable compatibility loading for older checkpoint key layouts.'
    )
    return parser.parse_args()

def load_checkpoint(checkpoint_path: str, legacy_compat: bool = False) -> Dict[str, Any]:
    """Load checkpoint in current format, optionally with legacy compatibility."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint_ext = os.path.splitext(checkpoint_path)[1].lower()

    if checkpoint_ext == ".safetensors":
        if safetensors_load_file is None:
            raise ImportError(
                "safetensors is required to load .safetensors checkpoints. "
                "Install with: pip install safetensors"
            )

        model_state_dict = safetensors_load_file(checkpoint_path, device="cpu")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, "config.json")
        thresholds_path = os.path.join(checkpoint_dir, "optimal_thresholds.json")

        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            print(f"Warning: no config.json found next to safetensors file: {config_path}")

        optimal_thresholds = {}
        if os.path.exists(thresholds_path):
            with open(thresholds_path, "r") as f:
                optimal_thresholds = json.load(f)

        checkpoint = {
            "model_state_dict": model_state_dict,
            "config": config,
            "optimal_thresholds": optimal_thresholds,
        }
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if legacy_compat and "model_state_dict" not in checkpoint and "state_dict" in checkpoint:
        checkpoint = {
            "model_state_dict": checkpoint["state_dict"],
            "config": checkpoint.get("config", {}),
            "epoch": checkpoint.get("epoch", 0),
            "val_metrics": checkpoint.get("val_metrics", {}),
        }

    required_keys = ["model_state_dict", "config"]
    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        raise ValueError(
            f"Checkpoint is missing required keys: {missing}. "
            "Please use a checkpoint produced by the current train.py pipeline."
        )

    return checkpoint


def _remap_legacy_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply lightweight key remapping for common legacy checkpoint layouts."""
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("resnet34.fc."):
            new_key = "fc." + new_key[len("resnet34.fc."):]
        elif new_key.startswith("resnet18.fc."):
            new_key = "fc." + new_key[len("resnet18.fc."):]
        elif new_key.startswith("resnet50.fc."):
            new_key = "fc." + new_key[len("resnet50.fc."):]
        remapped[new_key] = value
    return remapped


def _materialize_lazy_modules_from_state_dict(
    model: torch.nn.Module,
    state_dict: Dict[str, Any],
    dropout: float,
) -> None:
    """
    Materialize lazily-created modules (e.g., flattened_processor) before load_state_dict.
    """
    weight_key = "classifier.feature_extractor.flattened_processor.0.weight"
    if (
        weight_key in state_dict
        and hasattr(model, "classifier")
        and hasattr(model.classifier, "feature_extractor")
        and not hasattr(model.classifier.feature_extractor, "flattened_processor")
    ):
        linear_weight = state_dict[weight_key]
        out_dim, in_dim = linear_weight.shape
        model.classifier.feature_extractor.flattened_processor = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(out_dim),
        )


def create_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    biomarker_config: FlexibleBiomarkerConfig,
    legacy_compat: bool = False
) -> Tuple[torch.nn.Module, ExperimentConfig]:
    """Create model + config from checkpoint."""
    config_dict = checkpoint["config"]

    # Create experiment config with all required parameters
    config = ExperimentConfig(
        model=config_dict.get('model', 'ResNet-18'),
        loss_function=config_dict.get('loss_function', 'CE'),
        must_include=config_dict.get('must_include', True),
        learning_rate=config_dict.get('learning_rate', 1e-3),
        batch_size=config_dict.get('batch_size', 16),
        weight_decay=config_dict.get('weight_decay', 1e-5),
        optimizer=config_dict.get('optimizer', 'AdamW'),
        scheduler=config_dict.get('scheduler', 'CosineAnnealing'),
        image_augmentations=config_dict.get('image_augmentations', DEFAULT_AUGMENTATIONS.copy()),
        dropout=config_dict.get('dropout', 0.1),
        loss_specific_params=config_dict.get('loss_specific_params', 'class_weights=inverse_frequency'),
        multi_target_strategy=config_dict.get('multi_target_strategy', 'Shared backbone + task-specific heads'),
        single_target_strategy=config_dict.get('single_target_strategy', ''),
        pretrained_weights=config_dict.get('pretrained_weights', 'ImageNet'),
        fine_tuning_strategy=config_dict.get('fine_tuning_strategy', 'full'),
        expected_gpu_memory=config_dict.get('expected_gpu_memory', '8-10GB'),
        architectural_family=config_dict.get('architectural_family', 'CNN'),
        class_weighting=config_dict.get('class_weighting', 'inverse_frequency'),
        sampling_strategy=config_dict.get('sampling_strategy', 'balanced_batch'),
        threshold_selection=config_dict.get('threshold_selection', 'F1_optimal')
    )
    single_target_strategy = config_dict.get('single_target_strategy', '')

    print(f"Creating model: {config.model}")
    print(f"Fine-tuning strategy: {config.fine_tuning_strategy}")
    if single_target_strategy:
        print(f"Single-target strategy: {single_target_strategy}")

    # Align optional target feature dimension with saved task head input if present.
    expected_head_dim = None
    for key, tensor in checkpoint['model_state_dict'].items():
        if '.task_heads.' in key and key.endswith('.weight'):
            expected_head_dim = tensor.shape[1]
            break

    # Create model using ModelFactory
    model = ModelFactory.create_model(
        architecture=config.model,
        num_classes=biomarker_config.total_output_size,
        pretrained_weights=config.pretrained_weights,
        fine_tuning_strategy=config.fine_tuning_strategy,
        dropout=config.dropout,
        biomarker_config=biomarker_config,
        single_target_strategy=single_target_strategy,
        single_target_output_dim=expected_head_dim
    )

    state_dict_to_load = checkpoint['model_state_dict']
    if legacy_compat:
        state_dict_to_load = _remap_legacy_state_dict_keys(state_dict_to_load)

    _materialize_lazy_modules_from_state_dict(
        model=model,
        state_dict=state_dict_to_load,
        dropout=config.dropout,
    )

    missing_keys, unexpected_keys = model.load_state_dict(state_dict_to_load, strict=False)
    if missing_keys or unexpected_keys:
        print("State dict loading warnings:")
        if missing_keys:
            print(f"  Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        print("Model loaded successfully despite key mismatches")
    else:
        print("Model state dict loaded perfectly!")

    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, config

def create_test_transforms(config: ExperimentConfig) -> transforms.Compose:
    """Create test transforms that match training preprocessing exactly"""
    
    # CRITICAL: Parse augmentation string to get the EXACT same settings as training
    from config.experiment_config import parse_augmentation_string
    aug_params = parse_augmentation_string(config.image_augmentations)
    
    print(f"Test preprocessing settings:")
    print(f"   Pretrained weights: {config.pretrained_weights}")
    print(f"   ImageNet normalization: {aug_params['imagenet_norm']}")
    print(f"   Image augmentations: {config.image_augmentations}")
    
    transform_list = [transforms.ToTensor()]
    
    # CRITICAL: Convert grayscale to 3-channel for pre-trained models (matches train.py)
    transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    
    # CRITICAL: Use the EXACT same normalization logic as train.py
    # Only apply normalization if aug_params['imagenet_norm'] is True
    if aug_params['imagenet_norm']:
        if config.pretrained_weights == "ImageNet":
            # Use ImageNet normalization for ImageNet pre-trained models
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ))
        elif config.pretrained_weights == "RadImageNet":
            # Use RadImageNet normalization (medical imaging specific)
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
        print(f"Normalization applied: {config.pretrained_weights} normalization")
    else:
        print(f"No normalization applied (imagenet_norm=False)")
    
    return transforms.Compose(transform_list)

def create_test_dataset(data_dir: str, biomarker_config: FlexibleBiomarkerConfig, 
                       config: ExperimentConfig, size: int = 256, only_pred: bool = False, 
                       test_csv: str = 'test.csv', batch_size: int = 16) -> DataLoader:
    """Create test dataset and dataloader with matching preprocessing"""
    
    # Create transforms that match training exactly
    transform = create_test_transforms(config)
    
    if only_pred:
        # Test dataset without labels
        test_dataset = PredictionDataset(data_dir, transforms=transform, size=size)
        print(f"Created test dataset with {len(test_dataset)} images (prediction only)")
    else:
        # Use unified classifier dataset with explicit CSV selection
        test_dataset = ClassifierDataset(
            data_dir, 
            biomarker_config, 
            transforms=transform, 
            size=size, 
            train=False,
            csv_file=test_csv
        )
        print(f"Created test dataset with {len(test_dataset)} samples")
    
    return DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

def process_predictions(predictions: torch.Tensor, biomarker_config: FlexibleBiomarkerConfig) -> Dict[str, Any]:
    """Process raw predictions into interpretable outputs"""
    
    results = {}
    tensor_layout = biomarker_config.get_tensor_layout()
    
    # Process each biomarker type
    for biomarker in biomarker_config.binary_biomarkers:
        layout = tensor_layout[biomarker.name]
        pred_slice = predictions[:, layout.start_idx:layout.end_idx]  # [B, 1]
        
        # Apply sigmoid for binary classification
        prob = torch.sigmoid(pred_slice).cpu().numpy().flatten()
        results[biomarker.name] = prob
    
    for biomarker in biomarker_config.multiclass_biomarkers:
        layout = tensor_layout[biomarker.name]
        pred_slice = predictions[:, layout.start_idx:layout.end_idx]  # [B, num_classes]
        
        # Apply softmax for multiclass classification
        prob = F.softmax(pred_slice, dim=1).cpu().numpy()
        pred_class = np.argmax(prob, axis=1)
        
        results[f"{biomarker.name}_probabilities"] = prob
        results[f"{biomarker.name}_predicted_class"] = pred_class
    
    for biomarker in biomarker_config.continuous_biomarkers:
        layout = tensor_layout[biomarker.name]
        pred_slice = predictions[:, layout.start_idx:layout.end_idx]  # [B, 1]
        
        # Denormalize continuous predictions
        raw_pred = pred_slice.cpu().numpy().flatten()
        denormalized_pred = []
        
        for val in raw_pred:
            denormalized_val = biomarker.denormalize(val)
            denormalized_pred.append(denormalized_val)
        
        denormalized_pred = np.array(denormalized_pred)
        
        results[biomarker.name] = denormalized_pred
    
    return results

def find_optimal_thresholds_on_validation(model: torch.nn.Module, biomarker_config: FlexibleBiomarkerConfig, 
                                         data_dir: str, config: ExperimentConfig, size: int = 256, batch_size: int = 16) -> Dict[str, float]:
    """Find optimal thresholds by running inference on validation set"""
    
    print("🎯 Finding optimal thresholds on validation set...")
    
    # Create validation dataset (use train=False to get val.csv)
    transform = create_test_transforms(config)
    val_dataset = ClassifierDataset(data_dir, biomarker_config, transforms=transform, size=size, train=False)
    
    val_dataloader = DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run inference on validation set
    all_predictions = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_dataloader, desc="Validation inference")):
            images = images.to(device)
            targets = targets.to(device)
            
            # Convert single channel to 3-channel for models expecting RGB (matches train.py validation)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            
            # Forward pass
            predictions = model(images)
            
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(targets.detach().cpu())
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Find optimal thresholds
    optimal_thresholds = {}
    tensor_layout = biomarker_config.get_tensor_layout()
    
    # Get threshold search parameters from biomarker config (matches training exactly)
    validation_config = biomarker_config.validation
    threshold_range = validation_config.get('threshold_search_range', [0.1, 0.9])
    threshold_steps = validation_config.get('threshold_search_steps', 9)
    optimization_metric = validation_config.get('optimization_metric', 'f1_score')
    fallback_threshold = validation_config.get('fallback_threshold', 0.5)
    
    print(f"🎯 Using threshold search: {threshold_steps} steps from {threshold_range[0]} to {threshold_range[1]}")
    print(f"🎯 Optimizing for: {optimization_metric}")
    
    # Convert to numpy
    predictions_np = all_predictions.numpy()
    targets_np = all_targets.numpy()
    
    for biomarker in biomarker_config.binary_biomarkers:
        layout = tensor_layout[biomarker.name]
        
        pred_logits = predictions_np[:, layout.start_idx]
        pred_probs = 1 / (1 + np.exp(-pred_logits))  # Sigmoid
        true_labels = targets_np[:, layout.start_idx].astype(int)
        
        # Skip if all labels are the same
        if len(np.unique(true_labels)) < 2:
            optimal_thresholds[biomarker.name] = fallback_threshold
            print(f"  {biomarker.name}: Using fallback threshold ({fallback_threshold}) - insufficient label diversity")
            continue
        
        # Find optimal threshold using the configured metric
        best_threshold = fallback_threshold
        best_score = 0.0
        
        # Use the EXACT same threshold search parameters as training
        for threshold in np.linspace(threshold_range[0], threshold_range[1], threshold_steps):
            pred_labels = (pred_probs > threshold).astype(int)
            
            # Calculate the optimization metric
            tp = np.sum((pred_labels == 1) & (true_labels == 1))
            fp = np.sum((pred_labels == 1) & (true_labels == 0))
            fn = np.sum((pred_labels == 0) & (true_labels == 1))
            tn = np.sum((pred_labels == 0) & (true_labels == 0))
            
            # Calculate metric based on configuration
            if optimization_metric == 'f1_score' and tp + fp > 0 and tp + fn > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                score = 2 * (precision * recall) / (precision + recall)
            elif optimization_metric == 'accuracy':
                score = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            elif optimization_metric == 'precision' and tp + fp > 0:
                score = tp / (tp + fp)
            elif optimization_metric == 'recall' and tp + fn > 0:
                score = tp / (tp + fn)
            elif optimization_metric == 'specificity' and tn + fp > 0:
                score = tn / (tn + fp)
            else:
                score = 0.0  # Fallback
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        optimal_thresholds[biomarker.name] = best_threshold
        print(f"  {biomarker.name}: threshold={best_threshold:.3f}, {optimization_metric}={best_score:.3f}")
    
    return optimal_thresholds

def bootstrap_metric_ci(y_true, y_pred, metric_fn, n_bootstraps=1000, ci=0.95, seed=42):
    """Calculate bootstrapped confidence intervals for a metric"""
    rng = np.random.RandomState(seed)
    scores = []
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        try:
            score = metric_fn(y_true[indices], y_pred[indices])
            if not np.isnan(score):
                scores.append(score)
        except (ValueError, ZeroDivisionError):
            continue
    
    if len(scores) < 10:  # Need minimum samples for reliable CI
        return np.nan, np.nan
    
    sorted_scores = np.sort(scores)
    lower = np.percentile(sorted_scores, ((1.0 - ci) / 2.0) * 100)
    upper = np.percentile(sorted_scores, (1 - (1.0 - ci) / 2.0) * 100)
    return lower, upper

def calculate_enhanced_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                             biomarker_config: FlexibleBiomarkerConfig, 
                             optimal_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
    """Calculate enhanced metrics with bootstrapped confidence intervals"""
    
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    all_metrics = {}
    tensor_layout = biomarker_config.get_tensor_layout()
    
    # Binary classification metrics
    for biomarker in biomarker_config.binary_biomarkers:
        layout = tensor_layout[biomarker.name]
        
        pred_logits = predictions[:, layout.start_idx]
        pred_probs = 1 / (1 + np.exp(-pred_logits))  # Sigmoid
        true_labels = targets[:, layout.start_idx].astype(int)
        
        # Skip if all labels are the same
        if len(np.unique(true_labels)) < 2:
            continue
        
        # Get optimal threshold
        threshold = optimal_thresholds.get(biomarker.name, 0.5) if optimal_thresholds else 0.5
        pred_labels = (pred_probs > threshold).astype(int)
        
        # Calculate metrics
        metrics = {}
        
        # AUROC (threshold-independent)
        try:
            auroc = roc_auc_score(true_labels, pred_probs)
            auroc_ci = bootstrap_metric_ci(true_labels, pred_probs, roc_auc_score)
            metrics['auroc'] = auroc
            metrics['auroc_ci'] = auroc_ci
        except (ValueError, ZeroDivisionError):
            metrics['auroc'] = np.nan
            metrics['auroc_ci'] = (np.nan, np.nan)
        
        # Confusion matrix components
        tp = np.sum((pred_labels == 1) & (true_labels == 1))
        tn = np.sum((pred_labels == 0) & (true_labels == 0))
        fp = np.sum((pred_labels == 1) & (true_labels == 0))
        fn = np.sum((pred_labels == 0) & (true_labels == 1))
        
        # Precision, Recall, Specificity, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Calculate confidence intervals for threshold-dependent metrics
        def precision_fn(y_true, y_pred):
            pred_binary = (y_pred > threshold).astype(int)
            tp = np.sum((pred_binary == 1) & (y_true == 1))
            fp = np.sum((pred_binary == 1) & (y_true == 0))
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        def recall_fn(y_true, y_pred):
            pred_binary = (y_pred > threshold).astype(int)
            tp = np.sum((pred_binary == 1) & (y_true == 1))
            fn = np.sum((pred_binary == 0) & (y_true == 1))
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        def specificity_fn(y_true, y_pred):
            pred_binary = (y_pred > threshold).astype(int)
            tn = np.sum((pred_binary == 0) & (y_true == 0))
            fp = np.sum((pred_binary == 1) & (y_true == 0))
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        def f1_fn(y_true, y_pred):
            pred_binary = (y_pred > threshold).astype(int)
            tp = np.sum((pred_binary == 1) & (y_true == 1))
            fp = np.sum((pred_binary == 1) & (y_true == 0))
            fn = np.sum((pred_binary == 0) & (y_true == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        def accuracy_fn(y_true, y_pred):
            pred_binary = (y_pred > threshold).astype(int)
            return (pred_binary == y_true).mean()
        
        # Calculate confidence intervals
        precision_ci = bootstrap_metric_ci(true_labels, pred_probs, precision_fn)
        recall_ci = bootstrap_metric_ci(true_labels, pred_probs, recall_fn)
        specificity_ci = bootstrap_metric_ci(true_labels, pred_probs, specificity_fn)
        f1_ci = bootstrap_metric_ci(true_labels, pred_probs, f1_fn)
        accuracy_ci = bootstrap_metric_ci(true_labels, pred_probs, accuracy_fn)
        
        # Store metrics
        metrics.update({
            'precision': precision,
            'precision_ci': precision_ci,
            'recall': recall,
            'recall_ci': recall_ci,
            'specificity': specificity,
            'specificity_ci': specificity_ci,
            'f1_score': f1,
            'f1_score_ci': f1_ci,
            'accuracy': accuracy,
            'accuracy_ci': accuracy_ci,
            'threshold_used': threshold
        })
        
        all_metrics[biomarker.name] = metrics
    
    # Regression metrics
    for biomarker in biomarker_config.continuous_biomarkers:
        layout = tensor_layout[biomarker.name]
        
        pred_values_raw = predictions[:, layout.start_idx]
        true_values_raw = targets[:, layout.start_idx]
        
        # CRITICAL FIX: Do NOT apply sigmoid to regression predictions!
        # Regression models output raw continuous values, not probabilities
        # The model was trained without sigmoid activation for continuous outputs
        
        # Denormalize predictions and targets for proper metric calculation
        pred_values_denorm = np.array([biomarker.denormalize(val) for val in pred_values_raw])
        true_values_denorm = np.array([biomarker.denormalize(val) for val in true_values_raw])
        
        
        # Calculate metrics on denormalized values
        mae = mean_absolute_error(true_values_denorm, pred_values_denorm)
        mse = mean_squared_error(true_values_denorm, pred_values_denorm)
        r2 = r2_score(true_values_denorm, pred_values_denorm)
        
        # Calculate confidence intervals on denormalized values
        def mae_fn(y_true, y_pred):
            return mean_absolute_error(y_true, y_pred)
        
        def mse_fn(y_true, y_pred):
            return mean_squared_error(y_true, y_pred)
        
        def r2_fn(y_true, y_pred):
            return r2_score(y_true, y_pred)
        
        mae_ci = bootstrap_metric_ci(true_values_denorm, pred_values_denorm, mae_fn)
        mse_ci = bootstrap_metric_ci(true_values_denorm, pred_values_denorm, mse_fn)
        r2_ci = bootstrap_metric_ci(true_values_denorm, pred_values_denorm, r2_fn)
        
        all_metrics[biomarker.name] = {
            'mae': mae,
            'mae_ci': mae_ci,
            'mse': mse,
            'mse_ci': mse_ci,
            'r2_score': r2,
            'r2_score_ci': r2_ci
        }
    
    return all_metrics

def run_inference(model: torch.nn.Module, test_dataloader: DataLoader, 
                 biomarker_config: FlexibleBiomarkerConfig, optimal_thresholds: Dict[str, float] = None,
                 only_pred: bool = False) -> Dict[str, Any]:
    """Run inference on test set"""
    
    all_results = {}
    all_targets = {}
    all_predictions = []
    study_ids = []
    
    print("Running inference...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_dataloader)):
            if only_pred:
                images = batch_data
                batch_study_ids = [test_dataloader.dataset.at(batch_idx * args.batch_size + i) 
                                 for i in range(len(images))]
            else:
                images, targets = batch_data
                targets = targets.to(device)
                batch_study_ids = [test_dataloader.dataset.at(batch_idx * args.batch_size + i) 
                                 for i in range(len(images))]
                
                # Store targets for metrics calculation
                for i, target in enumerate(targets):
                    study_id = batch_study_ids[i]
                    all_targets[study_id] = target.cpu().numpy()
            
            images = images.to(device)
            
            # Convert single channel to 3-channel for models expecting RGB (matches train.py validation)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            
            # Forward pass
            predictions = model(images)
            all_predictions.append(predictions.cpu())
            study_ids.extend(batch_study_ids)
    
    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    
    # Process predictions
    processed_results = process_predictions(all_predictions, biomarker_config)
    
    # Add study IDs
    processed_results['STUDY_ID'] = study_ids
    
    # Calculate metrics if ground truth available
    if not only_pred and all_targets:
        print("Calculating metrics...")
        
        # Convert targets to tensor format - CRITICAL: ensure predictions and targets are aligned
        target_tensors = []
        prediction_indices = []
        for idx, study_id in enumerate(study_ids):
            if study_id in all_targets:
                target_tensors.append(torch.from_numpy(all_targets[study_id]))
                prediction_indices.append(idx)
        
        if target_tensors:
            target_tensor = torch.stack(target_tensors).to(device)
            # Only use predictions that have corresponding targets
            aligned_predictions = all_predictions[prediction_indices].to(device)
            metrics = calculate_enhanced_metrics(aligned_predictions, target_tensor, biomarker_config, optimal_thresholds)
            processed_results['metrics'] = metrics
    
    return processed_results

def save_results(results: Dict[str, Any], output_dir: str, biomarker_config: FlexibleBiomarkerConfig):
    """Save results to files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions CSV
    if args.save_predictions:
        # Create DataFrame from results
        df_data = {}
        
        # Add study IDs
        df_data['STUDY_ID'] = results['STUDY_ID']
        
        # Add predictions for each biomarker
        for biomarker in biomarker_config.binary_biomarkers:
            df_data[biomarker.name] = results[biomarker.name]
        
        for biomarker in biomarker_config.multiclass_biomarkers:
            df_data[f"{biomarker.name}_predicted_class"] = results[f"{biomarker.name}_predicted_class"]
            # Save probabilities as separate columns
            probs = results[f"{biomarker.name}_probabilities"]
            for i, class_name in enumerate(biomarker.classes):
                df_data[f"{biomarker.name}_{class_name}_prob"] = probs[:, i]
        
        for biomarker in biomarker_config.continuous_biomarkers:
            df_data[biomarker.name] = results[biomarker.name]
        
        df = pd.DataFrame(df_data)
        predictions_path = os.path.join(output_dir, 'predictions.csv')
        df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to: {predictions_path}")
    
    # Save metrics (only if --save_metrics flag is used)
    if 'metrics' in results and args.save_metrics:
        metrics = results['metrics']
        
        # Save detailed metrics JSON
        metrics_path = os.path.join(output_dir, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Detailed metrics saved to: {metrics_path}")
    
    # Print summary metrics (always show, regardless of save_metrics flag)
    if 'metrics' in results:
        metrics = results['metrics']
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        # Binary classification metrics
        if biomarker_config.binary_biomarkers:
            print("\nBinary Classification Metrics (with 95% CI):")
            for biomarker in biomarker_config.binary_biomarkers:
                if biomarker.name in metrics:
                    metric_data = metrics[biomarker.name]
                    print(f"  {biomarker.name}:")
                    
                    # AUROC
                    auroc = metric_data.get('auroc', np.nan)
                    auroc_ci = metric_data.get('auroc_ci', (np.nan, np.nan))
                    if not np.isnan(auroc):
                        print(f"    AUROC: {auroc:.4f} [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]")
                    
                    # Precision
                    precision = metric_data.get('precision', np.nan)
                    precision_ci = metric_data.get('precision_ci', (np.nan, np.nan))
                    if not np.isnan(precision):
                        print(f"    Precision: {precision:.4f} [{precision_ci[0]:.4f}, {precision_ci[1]:.4f}]")
                    
                    # Recall
                    recall = metric_data.get('recall', np.nan)
                    recall_ci = metric_data.get('recall_ci', (np.nan, np.nan))
                    if not np.isnan(recall):
                        print(f"    Recall: {recall:.4f} [{recall_ci[0]:.4f}, {recall_ci[1]:.4f}]")
                    
                    # Specificity
                    specificity = metric_data.get('specificity', np.nan)
                    specificity_ci = metric_data.get('specificity_ci', (np.nan, np.nan))
                    if not np.isnan(specificity):
                        print(f"    Specificity: {specificity:.4f} [{specificity_ci[0]:.4f}, {specificity_ci[1]:.4f}]")
                    
                    # F1-Score
                    f1 = metric_data.get('f1_score', np.nan)
                    f1_ci = metric_data.get('f1_score_ci', (np.nan, np.nan))
                    if not np.isnan(f1):
                        print(f"    F1-Score: {f1:.4f} [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
                    
                    # Accuracy
                    accuracy = metric_data.get('accuracy', np.nan)
                    accuracy_ci = metric_data.get('accuracy_ci', (np.nan, np.nan))
                    if not np.isnan(accuracy):
                        print(f"    Accuracy: {accuracy:.4f} [{accuracy_ci[0]:.4f}, {accuracy_ci[1]:.4f}]")
                    
                    # Threshold used
                    threshold = metric_data.get('threshold_used', 'N/A')
                    print(f"    Threshold used: {threshold}")
        
        # Multiclass classification metrics
        if biomarker_config.multiclass_biomarkers:
            print("\nMulticlass Classification Metrics:")
            for biomarker in biomarker_config.multiclass_biomarkers:
                if biomarker.name in metrics:
                    metric_data = metrics[biomarker.name]
                    print(f"  {biomarker.name}:")
                    print(f"    Accuracy: {metric_data.get('accuracy', 'N/A'):.4f}")
                    print(f"    F1-Score (macro): {metric_data.get('f1_score_macro', 'N/A'):.4f}")
        
        # Regression metrics
        if biomarker_config.continuous_biomarkers:
            print("\nRegression Metrics (with 95% CI):")
            for biomarker in biomarker_config.continuous_biomarkers:
                if biomarker.name in metrics:
                    metric_data = metrics[biomarker.name]
                    print(f"  {biomarker.name}:")
                    
                    # MAE
                    mae = metric_data.get('mae', np.nan)
                    mae_ci = metric_data.get('mae_ci', (np.nan, np.nan))
                    if not np.isnan(mae):
                        print(f"    MAE: {mae:.4f} [{mae_ci[0]:.4f}, {mae_ci[1]:.4f}]")
                    
                    # MSE
                    mse = metric_data.get('mse', np.nan)
                    mse_ci = metric_data.get('mse_ci', (np.nan, np.nan))
                    if not np.isnan(mse):
                        print(f"    MSE: {mse:.4f} [{mse_ci[0]:.4f}, {mse_ci[1]:.4f}]")
                    
                    # R²
                    r2 = metric_data.get('r2_score', np.nan)
                    r2_ci = metric_data.get('r2_score_ci', (np.nan, np.nan))
                    if not np.isnan(r2):
                        print(f"    R²: {r2:.4f} [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]")
        
        # Overall metrics
        if 'average_auroc' in metrics and metrics['average_auroc'] > 0:
            print(f"\nOverall Classification Performance:")
            print(f"  Average AUROC: {metrics['average_auroc']:.4f}")
            print(f"  Median AUROC: {metrics['median_auroc']:.4f}")
        
        if 'avg_regression_loss' in metrics:
            print(f"\nOverall Regression Performance:")
            print(f"  Average Regression Loss: {metrics['avg_regression_loss']:.4f}")
        
        print("="*60)

def main():
    global args
    args = arg_parse()
    
    print("="*60)
    print("FLEXIBLE MULTI-TASK TESTING")
    print("="*60)
    
    # Load biomarker configuration
    print(f"Loading biomarker configuration from: {args.biomarker_config}")
    biomarker_config = FlexibleBiomarkerConfig(args.biomarker_config)
    biomarker_config.print_summary()
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint_path, legacy_compat=args.legacy_checkpoint_compat)
    
    # Create model and get config
    model, config = create_model_from_checkpoint(
        checkpoint,
        biomarker_config,
        legacy_compat=args.legacy_checkpoint_compat
    )
    
    # Load optimal thresholds from checkpoint or find them on validation set
    optimal_thresholds = checkpoint.get('optimal_thresholds', {})
    if optimal_thresholds:
        print(f"Loaded optimal thresholds from checkpoint: {optimal_thresholds}")
    else:
        print("No optimal thresholds found in checkpoint.")
        if biomarker_config.binary_biomarkers:
            print("Finding optimal thresholds on validation set...")
            # Use validation data directory if specified, otherwise use same as test data
            val_data_dir = args.val_data_dir if args.val_data_dir else args.data_dir
            optimal_thresholds = find_optimal_thresholds_on_validation(
                model, biomarker_config, val_data_dir, config, args.size, args.batch_size
            )
        else:
            print("No binary biomarkers - skipping threshold optimization")
            optimal_thresholds = {}
    
    # Create test dataset with matching preprocessing
    test_dataloader = create_test_dataset(
        args.data_dir, 
        biomarker_config, 
        config,
        args.size, 
        args.only_pred,
        args.test_csv,
        args.batch_size
    )
    
    # Run inference
    results = run_inference(model, test_dataloader, biomarker_config, optimal_thresholds, args.only_pred)
    
    # Save results
    save_results(results, args.output_dir, biomarker_config)
    
    print(f"\nTesting completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
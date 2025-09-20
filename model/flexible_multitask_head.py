"""
Flexible Multi-Task Head and Loss Function
Adapts to any biomarker configuration without hardcoded assumptions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
from config.biomarker_config import FlexibleBiomarkerConfig, TensorLayout
from .single_target_strategies import (
    SingleTargetStrategy, 
    FeatureExtractor, 
    create_feature_extractor,
    get_strategy_from_csv
)


class FlexibleMultiTaskHead(nn.Module):
    """Flexible multi-task head that adapts to biomarker configuration"""
    
    def __init__(
        self, 
        input_dim: int, 
        biomarker_config: FlexibleBiomarkerConfig, 
        dropout: float = 0.1,
        single_target_strategy: Optional[Union[str, SingleTargetStrategy]] = None
    ):
        super().__init__()
        
        self.biomarker_config = biomarker_config
        self.tensor_layout = biomarker_config.get_tensor_layout()
        
        # Handle single-target strategy
        self.single_target_strategy = None
        self.feature_extractor = None
        
        if single_target_strategy is not None:
            if isinstance(single_target_strategy, str):
                self.single_target_strategy = get_strategy_from_csv(single_target_strategy)
            else:
                self.single_target_strategy = single_target_strategy
            
            # Create appropriate feature extractor
            self.feature_extractor = create_feature_extractor(
                self.single_target_strategy, 
                input_dim, 
                feature_dim=512, 
                dropout=dropout
            )
            
            # Use feature extractor output dimension
            processed_input_dim = self.feature_extractor.output_dim
        else:
            # Default behavior - use original input dimension
            processed_input_dim = input_dim
        
        # Shared feature processing (only if no single-target strategy is used)
        if self.single_target_strategy is None:
            self.shared_layers = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.BatchNorm1d(512)
            )
        else:
            # Skip shared layers when using single-target strategy
            self.shared_layers = nn.Identity()
            processed_input_dim = self.feature_extractor.output_dim
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        
        # Binary classification heads (one per biomarker)
        for biomarker in biomarker_config.binary_biomarkers:
            self.task_heads[f"binary_{biomarker.name}"] = nn.Linear(processed_input_dim, 1)
        
        # Multiclass classification heads
        for biomarker in biomarker_config.multiclass_biomarkers:
            num_classes = len(biomarker.classes)
            self.task_heads[f"multiclass_{biomarker.name}"] = nn.Linear(processed_input_dim, num_classes)
        
        # Regression heads (one per biomarker)
        for biomarker in biomarker_config.continuous_biomarkers:
            self.task_heads[f"continuous_{biomarker.name}"] = nn.Linear(processed_input_dim, 1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input features [batch_size, input_dim] or [batch_size, C, H, W] for spatial features
            
        Returns:
            Concatenated outputs [batch_size, total_output_size]
        """
        # Apply single-target strategy feature extraction if specified
        if self.single_target_strategy is not None and self.feature_extractor is not None:
            shared_features = self.feature_extractor(x)
        else:
            shared_features = self.shared_layers(x)
        
        outputs = []
        
        # Binary outputs
        for biomarker in self.biomarker_config.binary_biomarkers:
            head_name = f"binary_{biomarker.name}"
            binary_out = self.task_heads[head_name](shared_features)  # [B, 1]
            outputs.append(binary_out)
        
        # Multiclass outputs
        for biomarker in self.biomarker_config.multiclass_biomarkers:
            head_name = f"multiclass_{biomarker.name}"
            multiclass_out = self.task_heads[head_name](shared_features)  # [B, num_classes]
            outputs.append(multiclass_out)
        
        # Regression outputs
        for biomarker in self.biomarker_config.continuous_biomarkers:
            head_name = f"continuous_{biomarker.name}"
            regression_out = self.task_heads[head_name](shared_features)  # [B, 1]
            outputs.append(regression_out)
        
        # Concatenate all outputs
        return torch.cat(outputs, dim=1)


class LinearProbeMultiTaskHead(nn.Module):
    """
    True linear probe head - direct mapping from backbone features to task outputs
    No shared layers, minimal parameters, maximum interpretability
    """
    
    def __init__(
        self, 
        input_dim: int, 
        biomarker_config: FlexibleBiomarkerConfig, 
        dropout: float = 0.0,
        single_target_strategy: Optional[Union[str, SingleTargetStrategy]] = None
    ):
        super().__init__()
        
        self.biomarker_config = biomarker_config
        self.tensor_layout = biomarker_config.get_tensor_layout()
        
        # Handle single-target strategy
        self.single_target_strategy = None
        self.feature_extractor = None
        
        if single_target_strategy is not None:
            if isinstance(single_target_strategy, str):
                self.single_target_strategy = get_strategy_from_csv(single_target_strategy)
            else:
                self.single_target_strategy = single_target_strategy
            
            # Create appropriate feature extractor
            self.feature_extractor = create_feature_extractor(
                self.single_target_strategy, 
                input_dim, 
                feature_dim=input_dim,  # Keep same dimension for linear probe
                dropout=0.0  # No dropout for linear probe
            )
            
            # Use feature extractor output dimension
            processed_input_dim = self.feature_extractor.output_dim
        else:
            # Default behavior - use original input dimension
            processed_input_dim = input_dim
        
        # Optional minimal dropout (usually 0.0 for true linear probe)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Direct task-specific linear layers (no shared processing)
        self.task_heads = nn.ModuleDict()
        
        # Binary classification heads - direct from backbone
        for biomarker in biomarker_config.binary_biomarkers:
            self.task_heads[f"binary_{biomarker.name}"] = nn.Linear(processed_input_dim, 1)
        
        # Multiclass classification heads - direct from backbone  
        for biomarker in biomarker_config.multiclass_biomarkers:
            num_classes = len(biomarker.classes)
            self.task_heads[f"multiclass_{biomarker.name}"] = nn.Linear(processed_input_dim, num_classes)
        
        # Regression heads - direct from backbone
        for biomarker in biomarker_config.continuous_biomarkers:
            self.task_heads[f"continuous_{biomarker.name}"] = nn.Linear(processed_input_dim, 1)
        
        # Initialize weights for better convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize linear layer weights for stable training"""
        for name, module in self.task_heads.items():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Direct forward pass - no shared processing
        
        Args:
            x: Backbone features [batch_size, input_dim] or [batch_size, C, H, W] for spatial features
            
        Returns:
            Concatenated outputs [batch_size, total_output_size]
        """
        # Apply single-target strategy feature extraction if specified
        if self.single_target_strategy is not None and self.feature_extractor is not None:
            features = self.feature_extractor(x)
        else:
            # Optional dropout on backbone features (usually disabled)
            features = self.dropout(x)  # [batch_size, input_dim]
        
        outputs = []
        
        # Binary outputs - direct linear transformation
        for biomarker in self.biomarker_config.binary_biomarkers:
            head_name = f"binary_{biomarker.name}"
            binary_out = self.task_heads[head_name](features)  # [B, 1]
            outputs.append(binary_out)
        
        # Multiclass outputs - direct linear transformation
        for biomarker in self.biomarker_config.multiclass_biomarkers:
            head_name = f"multiclass_{biomarker.name}"
            multiclass_out = self.task_heads[head_name](features)  # [B, num_classes]
            outputs.append(multiclass_out)
        
        # Regression outputs - direct linear transformation
        for biomarker in self.biomarker_config.continuous_biomarkers:
            head_name = f"continuous_{biomarker.name}"
            regression_out = self.task_heads[head_name](features)  # [B, 1]
            outputs.append(regression_out)
        
        # Concatenate all outputs
        return torch.cat(outputs, dim=1)  # [batch_size, total_output_size]


class FlexibleMultiTaskLoss(nn.Module):
    """Flexible multi-task loss function that adapts to biomarker configuration"""
    
    def __init__(self, biomarker_config: FlexibleBiomarkerConfig, class_weights: Dict[str, float] = None):
        super().__init__()
        
        self.biomarker_config = biomarker_config
        self.tensor_layout = biomarker_config.get_tensor_layout()
        self.class_weights = class_weights or {}
        
        # Create weighted BCE losses for binary tasks
        self.binary_losses = nn.ModuleDict()
        for biomarker in biomarker_config.binary_biomarkers:
            weight = self.class_weights.get(biomarker.name, 1.0)
            if weight != 1.0:
                pos_weight = torch.tensor([weight], dtype=torch.float32)
                self.binary_losses[biomarker.name] = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.binary_losses[biomarker.name] = nn.BCEWithLogitsLoss()
        
        # Cross-entropy losses for multiclass tasks
        self.multiclass_losses = nn.ModuleDict()
        for biomarker in biomarker_config.multiclass_biomarkers:
            self.multiclass_losses[biomarker.name] = nn.CrossEntropyLoss()
        
        # MSE losses for regression tasks
        self.regression_losses = nn.ModuleDict()
        for biomarker in biomarker_config.continuous_biomarkers:
            self.regression_losses[biomarker.name] = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate multi-task loss
        
        Args:
            predictions: [batch_size, total_output_size]
            targets: [batch_size, total_output_size]
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        device = predictions.device
        
        # Move pos_weight tensors to correct device
        for biomarker_name, loss_fn in self.binary_losses.items():
            if hasattr(loss_fn, 'pos_weight') and loss_fn.pos_weight is not None:
                loss_fn.pos_weight = loss_fn.pos_weight.to(device)
        
        total_loss = 0.0
        loss_components = {}
        
        # Binary classification losses
        binary_losses = []
        for biomarker in self.biomarker_config.binary_biomarkers:
            layout = self.tensor_layout[biomarker.name]
            
            pred_slice = predictions[:, layout.start_idx:layout.end_idx].squeeze(-1)  # [B]
            target_slice = targets[:, layout.start_idx:layout.end_idx].squeeze(-1)  # [B]
            
            loss_fn = self.binary_losses[biomarker.name]
            binary_loss = loss_fn(pred_slice, target_slice)
            binary_losses.append(binary_loss)
            
            loss_components[f'binary_{biomarker.name}'] = binary_loss.item()
        
        if binary_losses:
            avg_binary_loss = torch.mean(torch.stack(binary_losses))
            total_loss += avg_binary_loss
            loss_components['avg_binary_loss'] = avg_binary_loss.item()
        
        # Multiclass classification losses
        multiclass_losses = []
        for biomarker in self.biomarker_config.multiclass_biomarkers:
            layout = self.tensor_layout[biomarker.name]
            
            pred_slice = predictions[:, layout.start_idx:layout.end_idx]  # [B, num_classes]
            target_slice = targets[:, layout.start_idx:layout.end_idx]  # [B, num_classes]
            
            # Convert one-hot targets to class indices
            target_indices = torch.argmax(target_slice, dim=1)  # [B]
            
            loss_fn = self.multiclass_losses[biomarker.name]
            multiclass_loss = loss_fn(pred_slice, target_indices)
            multiclass_losses.append(multiclass_loss)
            
            loss_components[f'multiclass_{biomarker.name}'] = multiclass_loss.item()
        
        if multiclass_losses:
            avg_multiclass_loss = torch.mean(torch.stack(multiclass_losses))
            total_loss += avg_multiclass_loss
            loss_components['avg_multiclass_loss'] = avg_multiclass_loss.item()
        
        # Regression losses
        regression_losses = []
        for biomarker in self.biomarker_config.continuous_biomarkers:
            layout = self.tensor_layout[biomarker.name]
            
            pred_slice = predictions[:, layout.start_idx:layout.end_idx].squeeze(-1)  # [B]
            target_slice = targets[:, layout.start_idx:layout.end_idx].squeeze(-1)  # [B]
            
            loss_fn = self.regression_losses[biomarker.name]
            regression_loss = loss_fn(pred_slice, target_slice)
            regression_losses.append(regression_loss)
            
            loss_components[f'regression_{biomarker.name}'] = regression_loss.item()
        
        if regression_losses:
            avg_regression_loss = torch.mean(torch.stack(regression_losses))
            total_loss += avg_regression_loss
            loss_components['avg_regression_loss'] = avg_regression_loss.item()
        
        loss_components['total_loss'] = total_loss.item()
        
        return total_loss, loss_components


class FlexibleMetricsCalculator:
    """Calculate comprehensive metrics for flexible multi-task learning"""
    
    def __init__(self, biomarker_config: FlexibleBiomarkerConfig):
        self.biomarker_config = biomarker_config
        self.tensor_layout = biomarker_config.get_tensor_layout()
        
        # Threshold optimization settings from config
        validation_config = biomarker_config.validation
        self.threshold_optimization = validation_config.get('threshold_optimization', False)
        self.optimization_metric = validation_config.get('optimization_metric', 'f1_score')
        self.per_biomarker_thresholds = validation_config.get('per_biomarker_thresholds', True)
        self.threshold_range = validation_config.get('threshold_search_range', [0.1, 0.9])
        self.threshold_steps = validation_config.get('threshold_search_steps', 81)
        self.fallback_threshold = validation_config.get('fallback_threshold', 0.5)
        
        # Store optimal thresholds per biomarker
        self.optimal_thresholds = {}
    
    def find_optimal_threshold(self, pred_probs: np.ndarray, true_labels: np.ndarray, 
                              metric: str = 'f1_score') -> Tuple[float, float]:
        """
        Find optimal threshold for a single biomarker based on specified metric
        
        Args:
            pred_probs: Predicted probabilities [N]
            true_labels: True binary labels [N]
            metric: Metric to optimize ('f1_score', 'sensitivity', 'specificity', etc.)
            
        Returns:
            best_threshold: Optimal threshold value
            best_score: Best metric score achieved
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        # Create threshold search space
        thresholds = np.linspace(self.threshold_range[0], self.threshold_range[1], self.threshold_steps)
        
        best_threshold = self.fallback_threshold
        best_score = 0.0
        
        for threshold in thresholds:
            pred_labels = (pred_probs > threshold).astype(int)
            
            try:
                if metric == 'f1_score':
                    score = f1_score(true_labels, pred_labels, zero_division=0.0)
                elif metric == 'precision':
                    score = precision_score(true_labels, pred_labels, zero_division=0.0)
                elif metric == 'recall' or metric == 'sensitivity':
                    score = recall_score(true_labels, pred_labels, zero_division=0.0)
                elif metric == 'specificity':
                    # Specificity = TN / (TN + FP)
                    tn = np.sum((pred_labels == 0) & (true_labels == 0))
                    fp = np.sum((pred_labels == 1) & (true_labels == 0))
                    score = tn / (tn + fp + 1e-8)
                elif metric == 'accuracy':
                    score = (pred_labels == true_labels).mean()
                else:
                    # Default to f1_score
                    score = f1_score(true_labels, pred_labels, zero_division=0.0)
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    
            except (ValueError, ZeroDivisionError):
                continue
        
        return best_threshold, best_score
    
    def optimize_thresholds(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Find optimal thresholds for all binary biomarkers
        
        Args:
            predictions: Model predictions [batch_size, total_output_size]
            targets: True targets [batch_size, total_output_size]
            
        Returns:
            Dictionary mapping biomarker names to optimal thresholds
        """
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        optimal_thresholds = {}
        
        for biomarker in self.biomarker_config.binary_biomarkers:
            layout = self.tensor_layout[biomarker.name]
            
            pred_logits = predictions[:, layout.start_idx]
            pred_probs = 1 / (1 + np.exp(-pred_logits))  # Sigmoid
            true_labels = targets[:, layout.start_idx].astype(int)
            
            # Skip if all labels are the same (no positive or negative examples)
            if len(np.unique(true_labels)) < 2:
                optimal_thresholds[biomarker.name] = self.fallback_threshold
                continue
            
            # Find optimal threshold
            best_threshold, best_score = self.find_optimal_threshold(
                pred_probs, true_labels, self.optimization_metric
            )
            
            optimal_thresholds[biomarker.name] = best_threshold
            
            # Log the optimization result
            print(f"  {biomarker.name}: threshold={best_threshold:.3f}, {self.optimization_metric}={best_score:.3f}")
        
        return optimal_thresholds
    
    def update_optimal_thresholds(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update optimal thresholds based on current predictions and targets"""
        if self.threshold_optimization:
            print("🎯 Optimizing thresholds...")
            self.optimal_thresholds = self.optimize_thresholds(predictions, targets)
        else:
            # Use fallback threshold for all biomarkers
            for biomarker in self.biomarker_config.binary_biomarkers:
                self.optimal_thresholds[biomarker.name] = self.fallback_threshold
    
    def calculate_binary_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, 
                                use_optimal_thresholds: bool = True) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for binary classification tasks"""
        metrics = {}
        
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        for biomarker in self.biomarker_config.binary_biomarkers:
            layout = self.tensor_layout[biomarker.name]
            
            pred_logits = predictions[:, layout.start_idx]
            pred_probs = 1 / (1 + np.exp(-pred_logits))  # Sigmoid
            true_labels = targets[:, layout.start_idx].astype(int)
            
            # AUROC (threshold-independent)
            try:
                from sklearn.metrics import roc_auc_score
                auroc = roc_auc_score(true_labels, pred_probs)
            except (ValueError, ImportError):
                auroc = 0.0
            
            # Get threshold for this biomarker
            if use_optimal_thresholds and biomarker.name in self.optimal_thresholds:
                threshold = self.optimal_thresholds[biomarker.name]
            else:
                threshold = self.fallback_threshold
            
            # Predictions with threshold
            pred_labels = (pred_probs > threshold).astype(int)
            
            # Basic metrics
            accuracy = (pred_labels == true_labels).mean()
            
            # Confusion matrix components
            true_positives = np.sum((pred_labels == 1) & (true_labels == 1))
            true_negatives = np.sum((pred_labels == 0) & (true_labels == 0))
            false_positives = np.sum((pred_labels == 1) & (true_labels == 0))
            false_negatives = np.sum((pred_labels == 0) & (true_labels == 1))
            
            # Sensitivity and Specificity
            sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
            
            # F1 Score
            try:
                from sklearn.metrics import f1_score
                f1 = f1_score(true_labels, pred_labels, zero_division=0.0)
            except ImportError:
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
                recall = sensitivity
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            metrics[biomarker.name] = {
                'auroc': float(auroc),
                'accuracy': float(accuracy),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'f1': float(f1),
                'threshold': float(threshold),  # Include threshold used
                'true_positives': int(true_positives),
                'true_negatives': int(true_negatives),
                'false_positives': int(false_positives),
                'false_negatives': int(false_negatives)
            }
        
        return metrics
    
    def calculate_multiclass_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, Dict[str, Any]]:
        """Calculate metrics for multiclass classification tasks"""
        metrics = {}
        
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        for biomarker in self.biomarker_config.multiclass_biomarkers:
            layout = self.tensor_layout[biomarker.name]
            
            pred_logits = predictions[:, layout.start_idx:layout.end_idx]
            # Numerically stable softmax
            max_logits = np.max(pred_logits, axis=1, keepdims=True)
            exp_logits = np.exp(pred_logits - max_logits)
            pred_probs = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-12)
            target_one_hot = targets[:, layout.start_idx:layout.end_idx]
            
            # Get predicted and true classes
            pred_classes = np.argmax(pred_probs, axis=1)
            true_classes = np.argmax(target_one_hot, axis=1)
            
            # Overall accuracy
            accuracy = (pred_classes == true_classes).mean()
            
            # Multi-class AUROC
            try:
                from sklearn.metrics import roc_auc_score
                auroc = roc_auc_score(target_one_hot, pred_probs, multi_class='ovr', average='macro')
            except (ValueError, ImportError):
                auroc = 0.0
            
            metrics[biomarker.name] = {
                'accuracy': float(accuracy),
                'auroc': float(auroc)
            }
        
        return metrics
    
    def calculate_regression_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for regression tasks"""
        metrics = {}
        
        # Convert to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        for biomarker in self.biomarker_config.continuous_biomarkers:
            layout = self.tensor_layout[biomarker.name]
            
            pred_values = predictions[:, layout.start_idx]
            true_values = targets[:, layout.start_idx]
            
            # Denormalize if needed
            if biomarker.normalization == "min_max":
                pred_denorm = pred_values * (biomarker.max_value - biomarker.min_value) + biomarker.min_value
                true_denorm = true_values * (biomarker.max_value - biomarker.min_value) + biomarker.min_value
            else:
                pred_denorm = pred_values
                true_denorm = true_values
            
            # Calculate metrics
            mse = np.mean((pred_denorm - true_denorm) ** 2)
            mae = np.mean(np.abs(pred_denorm - true_denorm))
            
            # R² score
            ss_res = np.sum((true_denorm - pred_denorm) ** 2)
            ss_tot = np.sum((true_denorm - np.mean(true_denorm)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            metrics[biomarker.name] = {
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2)
            }
        
        return metrics
    
    def calculate_all_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """Calculate all metrics"""
        all_metrics = {}
        
        # Binary metrics
        if self.biomarker_config.binary_biomarkers:
            binary_metrics = self.calculate_binary_metrics(predictions, targets)
            all_metrics.update(binary_metrics)
        
        # Multiclass metrics
        if self.biomarker_config.multiclass_biomarkers:
            multiclass_metrics = self.calculate_multiclass_metrics(predictions, targets)
            all_metrics.update(multiclass_metrics)
        
        # Regression metrics
        if self.biomarker_config.continuous_biomarkers:
            regression_metrics = self.calculate_regression_metrics(predictions, targets)
            all_metrics.update(regression_metrics)
        
        # Calculate both average and median AUROC for comprehensive monitoring
        auroc_values = []
        for biomarker_name, metrics in all_metrics.items():
            if isinstance(metrics, dict) and 'auroc' in metrics:
                auroc_values.append(metrics['auroc'])
        
        if auroc_values:
            all_metrics['average_auroc'] = float(np.mean(auroc_values))
            all_metrics['median_auroc'] = float(np.median(auroc_values))
        else:
            all_metrics['average_auroc'] = 0.0
            all_metrics['median_auroc'] = 0.0
        
        return all_metrics


if __name__ == "__main__":
    # Test the flexible components
    from config.biomarker_config import FlexibleBiomarkerConfig
    
    # Load configuration
    config = FlexibleBiomarkerConfig('config/biomarker_config_comorbidities.yaml')
    
    # Test multi-task head
    head = FlexibleMultiTaskHead(input_dim=512, biomarker_config=config)
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 512)
    output = head(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output size: {config.total_output_size}")
    
    # Test loss function
    criterion = FlexibleMultiTaskLoss(config)
    
    # Create dummy targets
    targets = torch.randn(batch_size, config.total_output_size)
    # Make binary targets actually binary
    for biomarker in config.binary_biomarkers:
        layout = config.get_tensor_layout()[biomarker.name]
        targets[:, layout.start_idx] = torch.randint(0, 2, (batch_size,)).float()
    
    loss, loss_dict = criterion(output, targets)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    # Test metrics
    metrics_calc = FlexibleMetricsCalculator(config)
    
    # Apply sigmoid to binary outputs for metrics calculation
    output_for_metrics = output.clone()
    metrics = metrics_calc.calculate_all_metrics(output_for_metrics, targets)
    
    print(f"Sample metrics: {list(metrics.keys())}")
    print(f"Average AUROC: {metrics['average_auroc']:.4f}")

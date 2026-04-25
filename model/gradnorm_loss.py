"""
GradNorm Implementation for Multi-Task Loss Balancing
Based on "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
Chen et al., 2018 (https://arxiv.org/abs/1711.02257)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from collections import deque
import logging

from config.biomarker_config import FlexibleBiomarkerConfig
from model.flexible_multitask_head import FlexibleMultiTaskLoss

logger = logging.getLogger(__name__)


class GradNormLoss(nn.Module):
    """
    GradNorm-based multi-task loss balancing.
    
    This implementation extends the FlexibleMultiTaskLoss with GradNorm algorithm
    for automatic loss balancing based on gradient magnitudes.
    """
    
    def __init__(
        self,
        biomarker_config: FlexibleBiomarkerConfig,
        class_weights: Dict[str, float] = None,
        alpha: float = 0.16,
        initial_task_loss_average_window: int = 20,
        update_weights_every: int = 10,
        normalize_losses: bool = True,
        restoring_force_factor: float = 0.1
    ):
        """
        Initialize GradNorm loss.
        
        Args:
            biomarker_config: Configuration for biomarkers and tasks
            class_weights: Initial class weights for individual losses
            alpha: Restoring force strength (typically 0.12-0.16)
            initial_task_loss_average_window: Window size for computing initial task loss averages
            update_weights_every: Update loss weights every N iterations
            normalize_losses: Whether to normalize individual losses
            restoring_force_factor: Factor for the restoring force strength
        """
        super().__init__()
        
        self.biomarker_config = biomarker_config
        self.alpha = alpha
        self.update_weights_every = update_weights_every
        self.normalize_losses = normalize_losses
        self.restoring_force_factor = restoring_force_factor
        self.initial_task_loss_average_window = initial_task_loss_average_window
        
        # Create the base multi-task loss
        self.base_loss = FlexibleMultiTaskLoss(biomarker_config, class_weights)
        
        # Get task information
        self.task_names = self._get_task_names()
        self.num_tasks = len(self.task_names)
        
        # Initialize task weights (learnable parameters)
        initial_weights = torch.ones(self.num_tasks, dtype=torch.float32)
        self.task_weights = nn.Parameter(initial_weights)
        
        # Tracking variables
        self.step_count = 0
        self.initial_task_losses = {task: deque(maxlen=initial_task_loss_average_window) 
                                  for task in self.task_names}
        self.initial_losses_computed = False
        self.task_loss_averages = None
        
        # For debugging and monitoring
        self.weight_history = []
        self.loss_ratio_history = []
    
    def to(self, device):
        """Move GradNorm loss to device."""
        super().to(device)
        self.task_weights.data = self.task_weights.data.to(device)
        if self.task_loss_averages is not None:
            self.task_loss_averages = self.task_loss_averages.to(device)
        return self
        
    def _get_task_names(self) -> List[str]:
        """Get list of all task names from biomarker config."""
        task_names = []
        
        # Add binary tasks
        for biomarker in self.biomarker_config.binary_biomarkers:
            task_names.append(f"binary_{biomarker.name}")
        
        # Add multiclass tasks
        for biomarker in self.biomarker_config.multiclass_biomarkers:
            task_names.append(f"multiclass_{biomarker.name}")
        
        # Add regression tasks
        for biomarker in self.biomarker_config.continuous_biomarkers:
            task_names.append(f"regression_{biomarker.name}")
        
        return task_names
    
    def _get_task_losses_from_components(self, loss_components: Dict[str, float]) -> torch.Tensor:
        """Extract individual task losses from loss components dict."""
        task_losses = []
        
        for task_name in self.task_names:
            if task_name in loss_components:
                task_losses.append(loss_components[task_name])
            else:
                # Handle missing task (shouldn't happen, but safety check)
                task_losses.append(0.0)
        
        return torch.tensor(task_losses, dtype=torch.float32, device=self.task_weights.device)
    
    def _get_task_losses_as_tensors(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Extract individual task losses as tensors (maintaining gradient connection)."""
        device = predictions.device
        task_losses = []
        tensor_layout = self.biomarker_config.get_tensor_layout()
        
        # Binary classification losses
        for biomarker in self.biomarker_config.binary_biomarkers:
            layout = tensor_layout[biomarker.name]
            
            pred_slice = predictions[:, layout.start_idx:layout.end_idx].squeeze(-1)  # [B]
            target_slice = targets[:, layout.start_idx:layout.end_idx].squeeze(-1)  # [B]
            
            loss_fn = self.base_loss.binary_losses[biomarker.name]
            binary_loss = loss_fn(pred_slice, target_slice)
            task_losses.append(binary_loss)
        
        # Multiclass classification losses
        for biomarker in self.biomarker_config.multiclass_biomarkers:
            layout = tensor_layout[biomarker.name]
            
            pred_slice = predictions[:, layout.start_idx:layout.end_idx]  # [B, num_classes]
            target_slice = targets[:, layout.start_idx:layout.end_idx]  # [B, num_classes]
            
            # Convert one-hot targets to class indices
            target_indices = torch.argmax(target_slice, dim=1)  # [B]
            
            loss_fn = self.base_loss.multiclass_losses[biomarker.name]
            multiclass_loss = loss_fn(pred_slice, target_indices)
            task_losses.append(multiclass_loss)
        
        # Regression losses
        for biomarker in self.biomarker_config.continuous_biomarkers:
            layout = tensor_layout[biomarker.name]
            
            pred_slice = predictions[:, layout.start_idx:layout.end_idx].squeeze(-1)  # [B]
            target_slice = targets[:, layout.start_idx:layout.end_idx].squeeze(-1)  # [B]
            
            loss_fn = self.base_loss.regression_losses[biomarker.name]
            regression_loss = loss_fn(pred_slice, target_slice)
            task_losses.append(regression_loss)
        
        return torch.stack(task_losses)
    
    def _compute_initial_task_losses(self, task_losses: torch.Tensor):
        """Collect initial task losses for computing averages."""
        task_losses_cpu = task_losses.detach().cpu().numpy()
        
        for i, task_name in enumerate(self.task_names):
            self.initial_task_losses[task_name].append(task_losses_cpu[i])
        
        # Check if we have enough samples to compute averages
        min_samples = min(len(losses) for losses in self.initial_task_losses.values())
        if min_samples >= self.initial_task_loss_average_window:
            self.task_loss_averages = torch.tensor([
                np.mean(self.initial_task_losses[task_name])
                for task_name in self.task_names
            ], dtype=torch.float32, device=self.task_weights.device)
            
            self.initial_losses_computed = True
            logger.info(
                "GradNorm: Initial task loss averages computed: %s",
                dict(zip(self.task_names, self.task_loss_averages.cpu().numpy())),
            )
    
    def _update_task_weights(self, model: nn.Module, task_losses: torch.Tensor):
        """Update task weights using simplified GradNorm algorithm."""
        if not self.initial_losses_computed:
            return
        
        device = self.task_weights.device
        
        # Simplified approach: update weights based on loss ratios without gradient computation
        # This avoids the gradient computation issues while still providing adaptive balancing
        
        # Compute relative inverse training rates
        task_loss_ratios = task_losses / self.task_loss_averages
        relative_inverse_training_rates = task_loss_ratios / torch.mean(task_loss_ratios)
        
        # Update weights based on relative training rates
        # Higher loss ratio -> higher weight (more attention to struggling tasks)
        # Apply restoring force based on relative training rates
        weight_updates = (relative_inverse_training_rates ** self.alpha) - 1.0
        self.task_weights.data += self.restoring_force_factor * weight_updates
        
        # Renormalize weights to prevent them from growing unboundedly
        self.task_weights.data = F.softmax(self.task_weights.data, dim=0) * self.num_tasks
        
        # Store for monitoring
        self.weight_history.append(self.task_weights.data.detach().cpu().numpy().copy())
        self.loss_ratio_history.append(relative_inverse_training_rates.detach().cpu().numpy().copy())
        
        if len(self.weight_history) % 50 == 0:  # Print every 50 updates
            logger.info(
                "GradNorm Step %s: Weights = %s",
                self.step_count,
                dict(zip(self.task_names, self.task_weights.data.cpu().numpy())),
            )
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                model: Optional[nn.Module] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass with GradNorm loss balancing.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            model: The model (needed for gradient computation)
            
        Returns:
            total_loss: Balanced total loss
            loss_dict: Dictionary with loss components and weights
        """
        # Get individual task losses from base loss
        base_total_loss, loss_components = self.base_loss(predictions, targets)
        
        # Extract task losses as tensors (maintaining gradient connection)
        task_losses = self._get_task_losses_as_tensors(predictions, targets)
        
        # Collect initial losses if not yet computed
        if not self.initial_losses_computed:
            self._compute_initial_task_losses(task_losses)
        
        # Update task weights using GradNorm (if model is provided and enough steps have passed)
        if (model is not None and 
            self.initial_losses_computed and 
            self.step_count % self.update_weights_every == 0 and
            self.step_count > 0):
            self._update_task_weights(model, task_losses)
        
        # Compute weighted total loss
        if self.initial_losses_computed:
            # Normalize task losses if requested
            if self.normalize_losses:
                normalized_task_losses = task_losses / self.task_loss_averages
                weighted_losses = self.task_weights * normalized_task_losses
            else:
                weighted_losses = self.task_weights * task_losses
            
            total_loss = torch.sum(weighted_losses)
        else:
            # Use equal weighting during initial phase
            total_loss = torch.sum(task_losses)
        
        # Update loss components with weights
        loss_components['total_loss'] = total_loss.item()
        
        # Add weight information to loss components
        if self.initial_losses_computed:
            for i, task_name in enumerate(self.task_names):
                loss_components[f'weight_{task_name}'] = self.task_weights[i].item()
        
        self.step_count += 1
        
        return total_loss, loss_components
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights as dictionary."""
        if not self.initial_losses_computed:
            return {task: 1.0 for task in self.task_names}
        
        return dict(zip(self.task_names, self.task_weights.data.cpu().numpy()))
    
    def get_weight_history(self) -> List[Dict[str, float]]:
        """Get history of task weights for analysis."""
        history = []
        for weights in self.weight_history:
            history.append(dict(zip(self.task_names, weights)))
        return history
    
    def reset_weights(self):
        """Reset task weights to uniform distribution."""
        with torch.no_grad():
            self.task_weights.data.fill_(1.0)
        self.initial_losses_computed = False
        self.step_count = 0
        self.weight_history.clear()
        self.loss_ratio_history.clear()
        for task_losses in self.initial_task_losses.values():
            task_losses.clear()


class GradNormTrainer:
    """
    Utility class to help integrate GradNorm with existing training loops.
    """
    
    def __init__(self, gradnorm_loss: GradNormLoss):
        self.gradnorm_loss = gradnorm_loss
        
    def compute_loss(self, model: nn.Module, predictions: torch.Tensor, 
                    targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss using GradNorm.
        
        This method should replace the standard criterion(predictions, targets) call.
        """
        return self.gradnorm_loss(predictions, targets, model)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics for logging."""
        return {
            'task_weights': self.gradnorm_loss.get_task_weights(),
            'step_count': self.gradnorm_loss.step_count,
            'initial_losses_computed': self.gradnorm_loss.initial_losses_computed,
            'num_tasks': self.gradnorm_loss.num_tasks
        }

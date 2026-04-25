"""
Experiment Configuration System
Handles experiment parameters for training and evaluation
"""

import ast
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os


# Per-model defaults for pretrained_weights and single_target_strategy.
# These are sourced from the experimentation plan used in the published results.
# Users can override both via --pretrained_weights and --single_target_strategy CLI flags.
MODEL_DEFAULTS: Dict[str, Dict[str, str]] = {
    "ResNet-18":   {"pretrained_weights": "ImageNet", "single_target_strategy": "Direct classification head"},
    "ResNet-34":   {"pretrained_weights": "ImageNet", "single_target_strategy": "Direct classification head"},
    "DenseNet-121":{"pretrained_weights": "ImageNet", "single_target_strategy": "Direct classification head"},
    "ViT-Small (DINOv2)": {"pretrained_weights": "DINOv2 (self-supervised)",  "single_target_strategy": "CLS token classification"},
    "Swin Transformer-Base": {"pretrained_weights": "ImageNet-22K", "single_target_strategy": "CLS token classification"},
    "ResNet-50 (RadImageNet)": {"pretrained_weights": "RadImageNet", "single_target_strategy": "Direct classification head"},
}

DEFAULT_AUGMENTATION_PARAMS: Dict[str, Any] = {
    "rotation": 15,
    "horizontal_flip": True,
    "random_crop": True,
    "color_jitter": True,
    "brightness": 0.2,
    "contrast": 0.2,
    "imagenet_norm": True,
}

DEFAULT_AUGMENTATIONS = DEFAULT_AUGMENTATION_PARAMS.copy()


def get_model_defaults(model_name: str) -> Dict[str, str]:
    """
    Return the default pretrained_weights and single_target_strategy for a given model.
    Falls back to ImageNet weights and Direct classification head for unknown models.
    """
    return MODEL_DEFAULTS.get(
        model_name,
        {"pretrained_weights": "ImageNet", "single_target_strategy": "Direct classification head"}
    )


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""

    # Model configuration
    model: str
    loss_function: str
    must_include: bool
    learning_rate: List[float]
    batch_size: int
    weight_decay: float
    optimizer: str
    scheduler: str

    # Training configuration
    image_augmentations: Dict[str, Any]
    dropout: float
    loss_specific_params: str
    multi_target_strategy: str
    single_target_strategy: str
    pretrained_weights: str
    fine_tuning_strategy: str

    # System configuration
    expected_gpu_memory: str
    architectural_family: str
    class_weighting: str
    sampling_strategy: str
    threshold_selection: str

    # Additional configuration
    experiment_name: str = ""
    output_dir: str = ""

    # GradNorm configuration
    use_gradnorm: bool = False
    gradnorm_alpha: float = 0.16
    gradnorm_update_freq: int = 10

    def __post_init__(self):
        """Process configuration after initialization"""
        if isinstance(self.learning_rate, str):
            try:
                self.learning_rate = ast.literal_eval(self.learning_rate)
            except (ValueError, SyntaxError):
                try:
                    self.learning_rate = [float(self.learning_rate)]
                except ValueError:
                    self.learning_rate = [1e-4]

        if not isinstance(self.learning_rate, list):
            self.learning_rate = [self.learning_rate]

        self.image_augmentations = parse_augmentation_string(self.image_augmentations)

        if not self.experiment_name:
            self.experiment_name = self._generate_experiment_name()

    def _generate_experiment_name(self) -> str:
        """Generate a unique experiment name based on configuration"""
        import datetime

        model_clean = self.model.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        lr_str = f"lr{self.learning_rate[0]:.0e}" if len(self.learning_rate) == 1 else "lr_sweep"
        batch_str = f"bs{self.batch_size}"

        ft_suffix = ""
        if "frozen" in self.fine_tuning_strategy.lower() or "probe" in self.fine_tuning_strategy.lower():
            ft_suffix = "_frozen"
        elif "partial" in self.fine_tuning_strategy.lower():
            ft_suffix = "_partial"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_clean}_{lr_str}_{batch_str}{ft_suffix}_{timestamp}"

    def get_output_directory(self, base_dir: str) -> str:
        """Get the output directory for this experiment"""
        if self.output_dir:
            return self.output_dir
        return os.path.join(base_dir, self.experiment_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        learning_rate_value = self.learning_rate[0] if len(self.learning_rate) == 1 else self.learning_rate

        return {
            'model': self.model,
            'loss_function': self.loss_function,
            'must_include': self.must_include,
            'learning_rate': learning_rate_value,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'image_augmentations': self.image_augmentations,
            'dropout': self.dropout,
            'loss_specific_params': self.loss_specific_params,
            'multi_target_strategy': self.multi_target_strategy,
            'single_target_strategy': self.single_target_strategy,
            'pretrained_weights': self.pretrained_weights,
            'fine_tuning_strategy': self.fine_tuning_strategy,
            'expected_gpu_memory': self.expected_gpu_memory,
            'architectural_family': self.architectural_family,
            'class_weighting': self.class_weighting,
            'sampling_strategy': self.sampling_strategy,
            'threshold_selection': self.threshold_selection,
            'experiment_name': self.experiment_name,
        }


def parse_augmentation_string(aug_input: Any) -> Dict[str, Any]:
    """Normalize augmentation params into a validated parameter dictionary."""
    aug_params = DEFAULT_AUGMENTATION_PARAMS.copy()

    if aug_input is None:
        return aug_params

    if isinstance(aug_input, str):
        try:
            parsed = ast.literal_eval(aug_input)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                "image_augmentations must be a dict (or a dict-like string), "
                "not a free-form text description."
            ) from exc
        aug_input = parsed

    if not isinstance(aug_input, dict):
        raise ValueError("image_augmentations must be a dictionary of augmentation params.")

    aug_params.update(aug_input)

    # Enforce expected types
    aug_params["rotation"] = int(aug_params["rotation"])
    aug_params["horizontal_flip"] = bool(aug_params["horizontal_flip"])
    aug_params["random_crop"] = bool(aug_params["random_crop"])
    aug_params["color_jitter"] = bool(aug_params["color_jitter"])
    aug_params["brightness"] = float(aug_params["brightness"])
    aug_params["contrast"] = float(aug_params["contrast"])
    aug_params["imagenet_norm"] = bool(aug_params["imagenet_norm"])

    return aug_params


def create_optimizer(model_parameters, config: 'ExperimentConfig'):
    """Create optimizer based on configuration"""
    import torch.optim as optim

    if config.optimizer == 'AdamW':
        return optim.AdamW(
            model_parameters,
            lr=config.learning_rate[0],
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'Adam':
        return optim.Adam(
            model_parameters,
            lr=config.learning_rate[0],
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'SGD':
        return optim.SGD(
            model_parameters,
            lr=config.learning_rate[0],
            weight_decay=config.weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def create_scheduler(optimizer, config: 'ExperimentConfig', total_epochs: int):
    """Create learning rate scheduler based on configuration"""
    import torch.optim.lr_scheduler as lr_scheduler

    if config.scheduler == 'CosineAnnealing':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    elif config.scheduler == 'CosineAnnealingWarmRestarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif config.scheduler == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    elif config.scheduler == 'StepLR':
        return lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif config.scheduler == 'ExponentialLR':
        return lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")

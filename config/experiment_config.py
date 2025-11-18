"""
Experiment Configuration System
Handles loading and parsing of experiment parameters from CSV
"""

import pandas as pd
import ast
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os


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
    image_augmentations: str
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
        # Parse learning rates if they're in string format
        if isinstance(self.learning_rate, str):
            try:
                self.learning_rate = ast.literal_eval(self.learning_rate)
            except (ValueError, SyntaxError):
                # If parsing fails, try to extract single float
                try:
                    self.learning_rate = [float(self.learning_rate)]
                except ValueError:
                    self.learning_rate = [1e-4]  # Default fallback
        
        # Ensure learning_rate is always a list
        if not isinstance(self.learning_rate, list):
            self.learning_rate = [self.learning_rate]
        
        # Generate experiment name if not provided
        if not self.experiment_name:
            self.experiment_name = self._generate_experiment_name()
    
    def _generate_experiment_name(self) -> str:
        """Generate a unique experiment name based on configuration"""
        import datetime
        
        # Clean model name for filename
        model_clean = self.model.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        
        # Add key distinguishing features
        lr_str = f"lr{self.learning_rate[0]:.0e}" if len(self.learning_rate) == 1 else f"lr_sweep"
        batch_str = f"bs{self.batch_size}"
        
        # Add fine-tuning strategy if relevant
        ft_suffix = ""
        if "frozen" in self.fine_tuning_strategy.lower() or "probe" in self.fine_tuning_strategy.lower():
            ft_suffix = "_frozen"
        elif "partial" in self.fine_tuning_strategy.lower():
            ft_suffix = "_partial"
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{model_clean}_{lr_str}_{batch_str}{ft_suffix}_{timestamp}"
    
    def generate_lr_experiments(self) -> List['ExperimentConfig']:
        """Generate individual experiment configs for each learning rate"""
        if len(self.learning_rate) <= 1:
            return [self]
        
        experiments = []
        for lr in self.learning_rate:
            # Create a copy of the current config
            import copy
            config_copy = copy.deepcopy(self)
            
            # Set single learning rate
            config_copy.learning_rate = [lr]
            
            # Generate new experiment name with specific learning rate
            config_copy.experiment_name = config_copy._generate_experiment_name_with_lr(lr)
            
            experiments.append(config_copy)
        
        return experiments
    
    def _generate_experiment_name_with_lr(self, lr: float) -> str:
        """Generate experiment name for specific learning rate"""
        import datetime
        
        # Clean model name for filename
        model_clean = self.model.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        
        # Format learning rate nicely
        lr_str = f"lr{lr:.0e}"
        batch_str = f"bs{self.batch_size}"
        
        # Add fine-tuning strategy if relevant
        ft_suffix = ""
        if "frozen" in self.fine_tuning_strategy.lower() or "probe" in self.fine_tuning_strategy.lower():
            ft_suffix = "_frozen"
        elif "partial" in self.fine_tuning_strategy.lower():
            ft_suffix = "_partial"
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{model_clean}_{lr_str}_{batch_str}{ft_suffix}_{timestamp}"
    
    def get_output_directory(self, base_dir: str) -> str:
        """Get the output directory for this experiment"""
        if self.output_dir:
            return self.output_dir
        
        return os.path.join(base_dir, self.experiment_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        # For individual experiments (single learning rate), save as single value
        # For base configs (multiple learning rates), save as array
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
            'experiment_name': self.experiment_name
        }


class ExperimentConfigLoader:
    """Loads experiment configurations from CSV file"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        
    def load_all_configs(self) -> List[ExperimentConfig]:
        """Load all experiment configurations from CSV"""
        configs = []
        
        for idx, row in self.df.iterrows():
            # Skip empty rows
            if pd.isna(row['Model']) or row['Model'] == '':
                continue
                
            try:
                config = ExperimentConfig(
                    model=row['Model'],
                    loss_function=row['Loss Function'],
                    must_include=row['Must Include'] == 'Yes',
                    learning_rate=row['Learning Rate'],
                    batch_size=int(row['Batch Size']),
                    weight_decay=float(row['Weight Decay']),
                    optimizer=row['Optimizer'],
                    scheduler=row['Scheduler'],
                    image_augmentations=row['Image Augmentations'],
                    dropout=float(row['Dropout']),
                    loss_specific_params=row['Loss-Specific Params'],
                    multi_target_strategy=row['Multi_Target_Strategy'],
                    single_target_strategy=row['Single_Target_Strategy'],
                    pretrained_weights=row['Pretrained_Weights'],
                    fine_tuning_strategy=row['Fine_Tuning_Strategy'],
                    expected_gpu_memory=row['Expected_GPU_Memory'],
                    architectural_family=row['Architectural_Family'],
                    class_weighting=row['Class_Weighting'],
                    sampling_strategy=row['Sampling_Strategy'],
                    threshold_selection=row['Threshold_Selection'],
                    # GradNorm configuration (optional columns)
                    use_gradnorm=row.get('Use_GradNorm', 'No') == 'Yes' if 'Use_GradNorm' in row and not pd.isna(row['Use_GradNorm']) else False,
                    gradnorm_alpha=float(row['GradNorm_Alpha']) if 'GradNorm_Alpha' in row and not pd.isna(row['GradNorm_Alpha']) else 0.16,
                    gradnorm_update_freq=int(row['GradNorm_Update_Freq']) if 'GradNorm_Update_Freq' in row and not pd.isna(row['GradNorm_Update_Freq']) else 10
                )
                
                # Add Turing1 compatibility if column exists
                if 'Turing1' in row and not pd.isna(row['Turing1']):
                    config.turing1_compatible = row['Turing1'] == 'Yes'
                else:
                    config.turing1_compatible = True  # Default to compatible
                configs.append(config)
                
            except Exception as e:
                print(f"Error loading configuration for row {idx}: {e}")
                print(f"Row data: {row.to_dict()}")
                continue
        
        return configs
    
    def load_all_configs_with_lr_expansion(self) -> List[ExperimentConfig]:
        """Load all configurations and expand learning rate hyperparameter search"""
        base_configs = self.load_all_configs()
        expanded_configs = []
        
        for config in base_configs:
            # Expand each config into individual learning rate experiments
            lr_experiments = config.generate_lr_experiments()
            expanded_configs.extend(lr_experiments)
        
        return expanded_configs
    
    def load_must_include_configs_with_lr_expansion(self) -> List[ExperimentConfig]:
        """Load must-include configurations and expand learning rate hyperparameter search"""
        base_configs = self.load_must_include_configs()
        expanded_configs = []
        
        for config in base_configs:
            # Expand each config into individual learning rate experiments
            lr_experiments = config.generate_lr_experiments()
            expanded_configs.extend(lr_experiments)
        
        return expanded_configs
    
    def load_turing1_compatible_configs(self) -> List[ExperimentConfig]:
        """Load only configurations compatible with Turing1 GPUs"""
        all_configs = self.load_all_configs()
        return [config for config in all_configs if getattr(config, 'turing1_compatible', True)]
    
    def load_turing1_must_include_configs(self) -> List[ExperimentConfig]:
        """Load must-include configurations compatible with Turing1 GPUs"""
        must_include_configs = self.load_must_include_configs()
        return [config for config in must_include_configs if getattr(config, 'turing1_compatible', True)]
    
    def load_turing1_must_include_with_lr_expansion(self) -> List[ExperimentConfig]:
        """Load must-include Turing1-compatible configs with learning rate expansion"""
        base_configs = self.load_turing1_must_include_configs()
        expanded_configs = []
        
        for config in base_configs:
            lr_experiments = config.generate_lr_experiments()
            expanded_configs.extend(lr_experiments)
        
        return expanded_configs
    
    def load_must_include_configs(self) -> List[ExperimentConfig]:
        """Load only configurations marked as 'Must Include'"""
        all_configs = self.load_all_configs()
        return [config for config in all_configs if config.must_include]
    
    def load_config_by_model(self, model_name: str) -> Optional[ExperimentConfig]:
        """Load configuration for a specific model"""
        all_configs = self.load_all_configs()
        for config in all_configs:
            if config.model == model_name:
                return config
        return None
    
    def get_available_models(self) -> List[str]:
        """Get list of all available model names"""
        all_configs = self.load_all_configs()
        return [config.model for config in all_configs]
    
    def get_models_by_family(self, family: str) -> List[ExperimentConfig]:
        """Get all models from a specific architectural family"""
        all_configs = self.load_all_configs()
        return [config for config in all_configs if config.architectural_family == family]
    
    def get_memory_requirements(self) -> Dict[str, str]:
        """Get memory requirements for all models"""
        all_configs = self.load_all_configs()
        return {config.model: config.expected_gpu_memory for config in all_configs}


def parse_augmentation_string(aug_string: str) -> Dict[str, Any]:
    """Parse image augmentation string into parameters"""
    aug_params = {
        'rotation': 15,
        'horizontal_flip': True,
        'random_crop': True,
        'color_jitter': True,
        'brightness': 0.2,
        'contrast': 0.2,
        'imagenet_norm': True
    }
    
    # Parse rotation
    if 'rotation' in aug_string:
        import re
        rotation_match = re.search(r'rotation \(±(\d+)°\)', aug_string)
        if rotation_match:
            aug_params['rotation'] = int(rotation_match.group(1))
    
    # Parse brightness and contrast
    if 'brightness±' in aug_string:
        brightness_match = re.search(r'brightness±([\d.]+)', aug_string)
        if brightness_match:
            aug_params['brightness'] = float(brightness_match.group(1))
    
    if 'contrast±' in aug_string:
        contrast_match = re.search(r'contrast±([\d.]+)', aug_string)
        if contrast_match:
            aug_params['contrast'] = float(contrast_match.group(1))
    
    # Check for specific augmentations
    aug_params['horizontal_flip'] = 'horizontal flip' in aug_string
    aug_params['random_crop'] = 'random crop' in aug_string
    aug_params['color_jitter'] = 'color jitter' in aug_string
    aug_params['imagenet_norm'] = 'ImageNet normalization' in aug_string
    
    return aug_params


def create_optimizer(model_parameters, config: ExperimentConfig):
    """Create optimizer based on configuration"""
    import torch.optim as optim
    
    if config.optimizer == 'AdamW':
        return optim.AdamW(
            model_parameters,
            lr=config.learning_rate[0],  # Use first LR for initial setup
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


def create_scheduler(optimizer, config: ExperimentConfig, total_epochs: int):
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


if __name__ == "__main__":
    # Test configuration loading
    csv_path = "/lfs/skampere2/0/mahmedc/Comorbidities-Detection/CT-Disease-Detection/experimentation_plan_simplified.csv"
    
    if os.path.exists(csv_path):
        loader = ExperimentConfigLoader(csv_path)
        
        # Load all configurations
        all_configs = loader.load_all_configs()
        print(f"Loaded {len(all_configs)} configurations")
        
        # Load must-include configurations
        must_include = loader.load_must_include_configs()
        print(f"Must include: {len(must_include)} configurations")
        
        # Print first few configurations
        for i, config in enumerate(must_include[:3]):
            print(f"\nConfig {i+1}:")
            print(f"  Model: {config.model}")
            print(f"  Learning rates: {config.learning_rate}")
            print(f"  Experiment name: {config.experiment_name}")
            print(f"  Expected memory: {config.expected_gpu_memory}")
    else:
        print(f"CSV file not found at {csv_path}")

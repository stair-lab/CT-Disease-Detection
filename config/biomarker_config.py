"""
Flexible Biomarker Configuration System
Supports dynamic task configuration without hardcoded assumptions
"""

import yaml
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class BinaryBiomarker:
    """Configuration for a binary classification task"""
    name: str
    description: str
    positive_class: str
    negative_class: str = "ABSENT"
    class_weight: Optional[float] = None


@dataclass  
class MulticlassBiomarker:
    """Configuration for a multiclass classification task"""
    name: str
    description: str
    classes: List[str]
    class_weights: Optional[Dict[str, float]] = None


@dataclass
class ContinuousBiomarker:
    """Configuration for a regression task"""
    name: str
    description: str
    min_value: float
    max_value: float
    normalization: str = "min_max"  # "min_max", "z_score", or "none"

    def __post_init__(self):
        """Validate continuous biomarker configuration."""
        if self.max_value <= self.min_value:
            raise ValueError(
                f"Invalid range for {self.name}: max_value ({self.max_value}) "
                f"must be greater than min_value ({self.min_value})"
            )
        if self.normalization not in {"min_max", "z_score", "none"}:
            raise ValueError(
                f"Unsupported normalization '{self.normalization}' for {self.name}. "
                "Expected one of: min_max, z_score, none."
            )
    
    def normalize(self, value: float) -> float:
        """Normalize a continuous value based on the configured normalization method"""
        if self.normalization == "min_max":
            # Min-max normalization to [0, 1]
            return (value - self.min_value) / (self.max_value - self.min_value)
        elif self.normalization == "z_score":
            # Z-score normalization (would need mean and std, using min_max for now)
            return (value - self.min_value) / (self.max_value - self.min_value)
        elif self.normalization == "none":
            # No normalization
            return value
        else:
            # Default to min_max
            return (value - self.min_value) / (self.max_value - self.min_value)
    
    def denormalize(self, normalized_value: float) -> float:
        """Denormalize a normalized value back to original scale"""
        if self.normalization == "min_max":
            # Reverse min-max normalization from [0, 1] to original range
            return normalized_value * (self.max_value - self.min_value) + self.min_value
        elif self.normalization == "z_score":
            # Reverse z-score normalization (would need mean and std, using min_max for now)
            return normalized_value * (self.max_value - self.min_value) + self.min_value
        elif self.normalization == "none":
            # No normalization
            return normalized_value
        else:
            # Default to min_max
            return normalized_value * (self.max_value - self.min_value) + self.min_value


@dataclass
class TensorLayout:
    """Describes where each biomarker appears in the output tensor"""
    biomarker_name: str
    start_idx: int
    end_idx: int
    size: int
    task_type: str  # "binary", "multiclass", "continuous"


class FlexibleBiomarkerConfig:
    """Flexible biomarker configuration that adapts to any task structure"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.experiment_name: str = ""
        self.description: str = ""
        self.binary_biomarkers: List[BinaryBiomarker] = []
        self.multiclass_biomarkers: List[MulticlassBiomarker] = []
        self.continuous_biomarkers: List[ContinuousBiomarker] = []
        self.preprocessing: Dict[str, Any] = {}
        self.training: Dict[str, Any] = {}
        self.validation: Dict[str, Any] = {}
        
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML or JSON file"""
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path}")
        
        self._parse_config(config_data)
    
    def _parse_config(self, config_data: Dict[str, Any]):
        """Parse configuration data"""
        self.experiment_name = config_data.get('experiment_name', '')
        self.description = config_data.get('description', '')
        
        # Parse binary biomarkers
        binary_data = config_data.get('binary_biomarkers', [])
        self.binary_biomarkers = [
            BinaryBiomarker(
                name=b['name'],
                description=b['description'],
                positive_class=b['positive_class'],
                negative_class=b.get('negative_class', 'ABSENT'),
                class_weight=b.get('class_weight')
            )
            for b in binary_data
        ]
        
        # Parse multiclass biomarkers
        multiclass_data = config_data.get('multiclass_biomarkers', [])
        self.multiclass_biomarkers = [
            MulticlassBiomarker(
                name=m['name'],
                description=m['description'],
                classes=m['classes'],
                class_weights=m.get('class_weights')
            )
            for m in multiclass_data
        ]
        
        # Parse continuous biomarkers
        continuous_data = config_data.get('continuous_biomarkers', [])
        self.continuous_biomarkers = [
            ContinuousBiomarker(
                name=c['name'],
                description=c['description'],
                min_value=c['min_value'],
                max_value=c['max_value'],
                normalization=c.get('normalization', 'min_max')
            )
            for c in continuous_data
        ]
        
        # Parse other settings
        self.preprocessing = config_data.get('preprocessing', {})
        self.training = config_data.get('training', {})
        self.validation = config_data.get('validation', {})
    
    @property
    def num_binary_tasks(self) -> int:
        """Number of binary classification tasks"""
        return len(self.binary_biomarkers)
    
    @property
    def num_multiclass_tasks(self) -> int:
        """Number of multiclass classification tasks"""
        return len(self.multiclass_biomarkers)
    
    @property
    def num_continuous_tasks(self) -> int:
        """Number of regression tasks"""
        return len(self.continuous_biomarkers)
    
    @property
    def total_multiclass_outputs(self) -> int:
        """Total outputs needed for all multiclass tasks"""
        return sum(len(m.classes) for m in self.multiclass_biomarkers)
    
    @property
    def total_output_size(self) -> int:
        """Total size of output tensor"""
        return (self.num_binary_tasks + 
                self.total_multiclass_outputs + 
                self.num_continuous_tasks)
    
    def get_tensor_layout(self) -> Dict[str, TensorLayout]:
        """Get the layout of biomarkers in the output tensor"""
        layout = {}
        current_idx = 0
        
        # Binary biomarkers (1 output each)
        for biomarker in self.binary_biomarkers:
            layout[biomarker.name] = TensorLayout(
                biomarker_name=biomarker.name,
                start_idx=current_idx,
                end_idx=current_idx + 1,
                size=1,
                task_type="binary"
            )
            current_idx += 1
        
        # Multiclass biomarkers (n outputs each)
        for biomarker in self.multiclass_biomarkers:
            num_classes = len(biomarker.classes)
            layout[biomarker.name] = TensorLayout(
                biomarker_name=biomarker.name,
                start_idx=current_idx,
                end_idx=current_idx + num_classes,
                size=num_classes,
                task_type="multiclass"
            )
            current_idx += num_classes
        
        # Continuous biomarkers (1 output each)
        for biomarker in self.continuous_biomarkers:
            layout[biomarker.name] = TensorLayout(
                biomarker_name=biomarker.name,
                start_idx=current_idx,
                end_idx=current_idx + 1,
                size=1,
                task_type="continuous"
            )
            current_idx += 1
        
        return layout
    
    def get_all_biomarker_names(self) -> List[str]:
        """Get names of all biomarkers"""
        names = []
        names.extend([b.name for b in self.binary_biomarkers])
        names.extend([m.name for m in self.multiclass_biomarkers])
        names.extend([c.name for c in self.continuous_biomarkers])
        return names
    
    def validate_dataset_compatibility(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check if dataset has all required biomarker columns"""
        required_columns = self.get_all_biomarker_names()
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        is_compatible = len(missing_columns) == 0
        return is_compatible, missing_columns
    
    def prepare_targets_tensor(self, df: pd.DataFrame, indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Convert dataframe rows to target tensors for training
        
        Args:
            df: DataFrame with biomarker columns
            indices: Optional list of row indices to process (if None, process all)
            
        Returns:
            numpy array of shape [num_samples, total_output_size]
        """
        if indices is None:
            indices = list(range(len(df)))
        
        num_samples = len(indices)
        targets = np.zeros((num_samples, self.total_output_size))
        layout = self.get_tensor_layout()
        
        for i, row_idx in enumerate(indices):
            row = df.iloc[row_idx]
            
            # Process binary biomarkers
            for biomarker in self.binary_biomarkers:
                tensor_info = layout[biomarker.name]
                value = row[biomarker.name]
                
                # Convert to binary (1 if positive_class, 0 otherwise)
                if pd.isna(value):
                    binary_value = 0.0  # Default to negative class for missing values
                elif str(value).upper() == biomarker.positive_class.upper():
                    binary_value = 1.0
                elif str(value).upper() == "MALE" and biomarker.positive_class.upper() == "MALE":
                    binary_value = 1.0
                elif str(value).upper() == "FEMALE" and biomarker.positive_class.upper() == "MALE":
                    binary_value = 0.0
                else:
                    binary_value = 0.0
                
                targets[i, tensor_info.start_idx] = binary_value
            
            # Process multiclass biomarkers
            for biomarker in self.multiclass_biomarkers:
                tensor_info = layout[biomarker.name]
                value = str(row[biomarker.name]).upper()
                
                # Create one-hot encoding
                class_idx = -1
                for j, class_name in enumerate(biomarker.classes):
                    if value == class_name.upper():
                        class_idx = j
                        break
                
                if class_idx >= 0:
                    targets[i, tensor_info.start_idx + class_idx] = 1.0
                # If no match found, leave as zeros (unknown class)
            
            # Process continuous biomarkers
            for biomarker in self.continuous_biomarkers:
                tensor_info = layout[biomarker.name]
                value = row[biomarker.name]
                
                if pd.isna(value):
                    normalized_value = 0.0  # Default for missing values
                else:
                    normalized_value = biomarker.normalize(float(value))
                    if biomarker.normalization in {"min_max", "z_score"}:
                        # Keep normalized targets bounded for training stability.
                        normalized_value = float(np.clip(normalized_value, 0.0, 1.0))
                
                targets[i, tensor_info.start_idx] = normalized_value
        
        return targets
    
    def denormalize_continuous_predictions(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert normalized continuous predictions back to original scale"""
        layout = self.get_tensor_layout()
        denormalized = {}
        
        for biomarker in self.continuous_biomarkers:
            tensor_info = layout[biomarker.name]
            normalized_preds = predictions[:, tensor_info.start_idx]

            original_preds = np.array([biomarker.denormalize(v) for v in normalized_preds], dtype=np.float32)
            denormalized[biomarker.name] = original_preds
        
        return denormalized
    
    def save_to_file(self, file_path: str):
        """Save configuration to file"""
        config_data = {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'binary_biomarkers': [
                {
                    'name': b.name,
                    'description': b.description,
                    'positive_class': b.positive_class,
                    'negative_class': b.negative_class
                }
                for b in self.binary_biomarkers
            ],
            'multiclass_biomarkers': [
                {
                    'name': m.name,
                    'description': m.description,
                    'classes': m.classes
                }
                for m in self.multiclass_biomarkers
            ],
            'continuous_biomarkers': [
                {
                    'name': c.name,
                    'description': c.description,
                    'min_value': c.min_value,
                    'max_value': c.max_value,
                    'normalization': c.normalization
                }
                for c in self.continuous_biomarkers
            ],
            'preprocessing': self.preprocessing,
            'training': self.training,
            'validation': self.validation
        }
        
        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
            with open(file_path, 'w') as f:
                yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False, indent=2)
        elif file_path.endswith('.json'):
            with open(file_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def print_summary(self):
        """Print a summary of the configuration"""
        print(f"Experiment: {self.experiment_name}")
        print(f"Description: {self.description}")
        print(f"\nTask Configuration:")
        print(f"  Binary tasks: {self.num_binary_tasks}")
        print(f"  Multiclass tasks: {self.num_multiclass_tasks}")
        print(f"  Continuous tasks: {self.num_continuous_tasks}")
        print(f"  Total output size: {self.total_output_size}")
        
        print(f"\nBinary Biomarkers:")
        for b in self.binary_biomarkers:
            print(f"  - {b.name}: {b.description}")
        
        if self.multiclass_biomarkers:
            print(f"\nMulticlass Biomarkers:")
            for m in self.multiclass_biomarkers:
                print(f"  - {m.name}: {m.description} ({len(m.classes)} classes)")
        
        print(f"\nContinuous Biomarkers:")
        for c in self.continuous_biomarkers:
            print(f"  - {c.name}: {c.description} (range: {c.min_value}-{c.max_value})")
        
        print(f"\nTensor Layout:")
        layout = self.get_tensor_layout()
        for name, info in layout.items():
            print(f"  {name}: indices {info.start_idx}-{info.end_idx-1} (size: {info.size}, type: {info.task_type})")


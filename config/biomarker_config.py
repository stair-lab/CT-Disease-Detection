"""
Biomarker Configuration System
Allows flexible specification of binary, multiclass, and continuous biomarkers
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import yaml


@dataclass
class BinaryBiomarker:
    """Configuration for binary classification biomarker"""
    name: str
    positive_class: str = "PRESENT"
    negative_class: str = "ABSENT"
    class_weight: Optional[float] = None
    
    def __post_init__(self):
        if self.class_weight is None:
            self.class_weight = 1.0


@dataclass
class MultiClassBiomarker:
    """Configuration for multiclass biomarker"""
    name: str
    classes: List[str]  # e.g., ["ABSENT", "LOW", "MEDIUM", "HIGH"]
    class_weights: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.class_weights is None:
            self.class_weights = {cls: 1.0 for cls in self.classes}
        
        # Ensure all classes have weights
        for cls in self.classes:
            if cls not in self.class_weights:
                self.class_weights[cls] = 1.0
    
    @property
    def num_classes(self) -> int:
        return len(self.classes)
    
    def class_to_index(self, class_name: str) -> int:
        """Convert class name to index"""
        try:
            return self.classes.index(class_name)
        except ValueError:
            raise ValueError(f"Unknown class '{class_name}' for biomarker '{self.name}'. "
                           f"Available classes: {self.classes}")
    
    def index_to_class(self, index: int) -> str:
        """Convert index to class name"""
        if 0 <= index < len(self.classes):
            return self.classes[index]
        else:
            raise ValueError(f"Index {index} out of range for biomarker '{self.name}'. "
                           f"Valid range: 0-{len(self.classes)-1}")


@dataclass
class ContinuousBiomarker:
    """Configuration for continuous/regression biomarker"""
    name: str
    normalization_factor: float = 1.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    def normalize(self, value: float) -> float:
        """Normalize the continuous value"""
        return float(value) / self.normalization_factor
    
    def denormalize(self, normalized_value: float) -> float:
        """Denormalize the continuous value"""
        return normalized_value * self.normalization_factor


@dataclass
class BiomarkerConfig:
    """Complete biomarker configuration"""
    binary_biomarkers: List[BinaryBiomarker]
    multiclass_biomarkers: List[MultiClassBiomarker]
    continuous_biomarkers: List[ContinuousBiomarker]
    
    def __post_init__(self):
        # Validate no duplicate names
        all_names = []
        for biomarker in self.binary_biomarkers:
            all_names.append(biomarker.name)
        for biomarker in self.multiclass_biomarkers:
            all_names.append(biomarker.name)
        for biomarker in self.continuous_biomarkers:
            all_names.append(biomarker.name)
        
        if len(all_names) != len(set(all_names)):
            duplicates = [name for name in set(all_names) if all_names.count(name) > 1]
            raise ValueError(f"Duplicate biomarker names found: {duplicates}")
    
    @property
    def total_output_size(self) -> int:
        """Calculate total output tensor size"""
        size = 0
        size += len(self.binary_biomarkers)  # Binary tasks
        for mc_biomarker in self.multiclass_biomarkers:
            size += mc_biomarker.num_classes  # Multiclass one-hot
        size += len(self.continuous_biomarkers)  # Continuous tasks
        return size
    
    @property
    def all_biomarker_names(self) -> List[str]:
        """Get all biomarker names"""
        names = []
        names.extend([b.name for b in self.binary_biomarkers])
        names.extend([b.name for b in self.multiclass_biomarkers])
        names.extend([b.name for b in self.continuous_biomarkers])
        return names
    
    def get_tensor_layout(self) -> Dict[str, Dict[str, Any]]:
        """Get the layout of the output tensor"""
        layout = {}
        current_idx = 0
        
        # Binary biomarkers
        for biomarker in self.binary_biomarkers:
            layout[biomarker.name] = {
                'type': 'binary',
                'start_idx': current_idx,
                'end_idx': current_idx + 1,
                'size': 1
            }
            current_idx += 1
        
        # Multiclass biomarkers
        for biomarker in self.multiclass_biomarkers:
            layout[biomarker.name] = {
                'type': 'multiclass',
                'start_idx': current_idx,
                'end_idx': current_idx + biomarker.num_classes,
                'size': biomarker.num_classes,
                'classes': biomarker.classes
            }
            current_idx += biomarker.num_classes
        
        # Continuous biomarkers
        for biomarker in self.continuous_biomarkers:
            layout[biomarker.name] = {
                'type': 'continuous',
                'start_idx': current_idx,
                'end_idx': current_idx + 1,
                'size': 1,
                'normalization_factor': biomarker.normalization_factor
            }
            current_idx += 1
        
        return layout
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'binary_biomarkers': [
                {
                    'name': b.name,
                    'positive_class': b.positive_class,
                    'negative_class': b.negative_class,
                    'class_weight': b.class_weight
                } for b in self.binary_biomarkers
            ],
            'multiclass_biomarkers': [
                {
                    'name': b.name,
                    'classes': b.classes,
                    'class_weights': b.class_weights
                } for b in self.multiclass_biomarkers
            ],
            'continuous_biomarkers': [
                {
                    'name': b.name,
                    'normalization_factor': b.normalization_factor,
                    'min_value': b.min_value,
                    'max_value': b.max_value
                } for b in self.continuous_biomarkers
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiomarkerConfig':
        """Create from dictionary"""
        binary_biomarkers = [
            BinaryBiomarker(**b) for b in data.get('binary_biomarkers', [])
        ]
        multiclass_biomarkers = [
            MultiClassBiomarker(**b) for b in data.get('multiclass_biomarkers', [])
        ]
        continuous_biomarkers = [
            ContinuousBiomarker(**b) for b in data.get('continuous_biomarkers', [])
        ]
        
        return cls(binary_biomarkers, multiclass_biomarkers, continuous_biomarkers)
    
    def save_to_json(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def save_to_yaml(self, filepath: str):
        """Save configuration to YAML file"""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'BiomarkerConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def load_from_yaml(cls, filepath: str) -> 'BiomarkerConfig':
        """Load configuration from YAML file"""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


# Default configuration matching current system
def get_default_biomarker_config() -> BiomarkerConfig:
    """Get the default biomarker configuration matching current hardcoded setup"""
    
    binary_biomarkers = [
        BinaryBiomarker(name='HCC18'),
        BinaryBiomarker(name='HCC22'),
        BinaryBiomarker(name='HCC85'),
        BinaryBiomarker(name='HCC96'),
        BinaryBiomarker(name='HCC108'),
        BinaryBiomarker(name='HCC111'),
    ]
    
    multiclass_biomarkers = [
        MultiClassBiomarker(
            name='CalciumScoring_AbdominalAgatston',
            classes=['ABSENT', 'LOW', 'MEDIUM', 'HIGH']
        )
    ]
    
    continuous_biomarkers = [
        ContinuousBiomarker(name='AGE', normalization_factor=101.0),
        ContinuousBiomarker(name='RAF', normalization_factor=50.0)
    ]
    
    return BiomarkerConfig(binary_biomarkers, multiclass_biomarkers, continuous_biomarkers)


# Create example configurations
def create_example_configs():
    """Create example configuration files"""
    
    # Default configuration
    default_config = get_default_biomarker_config()
    default_config.save_to_yaml('biomarker_config_default.yaml')
    default_config.save_to_json('biomarker_config_default.json')
    
    # Custom example configuration
    custom_config = BiomarkerConfig(
        binary_biomarkers=[
            BinaryBiomarker(name='HCC18', class_weight=2.0),
            BinaryBiomarker(name='HCC22', class_weight=1.5),
            BinaryBiomarker(name='Diabetes', positive_class='YES', negative_class='NO'),
        ],
        multiclass_biomarkers=[
            MultiClassBiomarker(
                name='Severity',
                classes=['MILD', 'MODERATE', 'SEVERE'],
                class_weights={'MILD': 1.0, 'MODERATE': 1.5, 'SEVERE': 2.0}
            ),
            MultiClassBiomarker(
                name='Stage',
                classes=['I', 'II', 'III', 'IV']
            )
        ],
        continuous_biomarkers=[
            ContinuousBiomarker(name='AGE', normalization_factor=100.0, min_value=0, max_value=120),
            ContinuousBiomarker(name='BMI', normalization_factor=50.0, min_value=10, max_value=60),
            ContinuousBiomarker(name='BloodPressure', normalization_factor=200.0)
        ]
    )
    
    custom_config.save_to_yaml('biomarker_config_example.yaml')
    custom_config.save_to_json('biomarker_config_example.json')
    
    print("Example configuration files created:")
    print("- biomarker_config_default.yaml")
    print("- biomarker_config_default.json")
    print("- biomarker_config_example.yaml")
    print("- biomarker_config_example.json")


if __name__ == "__main__":
    # Test the configuration system
    config = get_default_biomarker_config()
    
    print("Default Biomarker Configuration:")
    print(f"Total output size: {config.total_output_size}")
    print(f"All biomarkers: {config.all_biomarker_names}")
    
    print("\nTensor Layout:")
    layout = config.get_tensor_layout()
    for name, info in layout.items():
        print(f"  {name}: {info}")
    
    # Create example files
    create_example_configs()



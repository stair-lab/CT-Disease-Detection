"""
Flexible Multi-Task Testing Script
Supports any biomarker configuration and model architecture
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import json
import yaml
from typing import Dict, Any, List, Tuple

from dataset import ClassifierDataset
from test_dataset import TestDataset
from model.model_factory import ModelFactory
from model.flexible_multitask_head import FlexibleMetricsCalculator
from config.biomarker_config import FlexibleBiomarkerConfig
from config.experiment_config import ExperimentConfig
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomCSVDataset(Dataset):
    """Custom dataset that can load any CSV file"""
    
    def __init__(self, data_path, biomarker_config, transforms=None, size=256, csv_file='test.csv'):
        """
        Initialize dataset with custom CSV file
        
        Args:
            data_path: Path to data directory
            biomarker_config: FlexibleBiomarkerConfig object
            transforms: Image transforms
            size: Image size
            csv_file: Name of CSV file to load (e.g., 'test.csv', 'val.csv', 'train.csv')
        """
        if not os.path.exists(data_path):
            raise IOError(f'Path given for CustomCSVDataset {data_path} does not exist...')
        
        self.data_path = data_path
        self.size = size
        self.biomarker_config = biomarker_config
        self.transforms = transforms
        
        # Load the specified CSV file
        csv_path = os.path.join(data_path, csv_file)
        if not os.path.exists(csv_path):
            raise IOError(f'CSV file {csv_path} does not exist...')
        
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples from {csv_file}")
        
        # Handle RAF column if it doesn't exist
        if 'RAF' not in self.df.columns:
            self.df['RAF'] = 0
        
        # Apply age filtering for HIPAA compliance (only if AGE is a biomarker being tested)
        self.df = self._filter_age_records()
        
        # Get tensor layout for efficient indexing
        self.tensor_layout = self.biomarker_config.get_tensor_layout()
        
        # Pre-compute all target tensors for efficient access
        self.targets = self._prepare_all_targets()
        
        print(f"Biomarkers configured: {self.biomarker_config.get_all_biomarker_names()}")
        print(f"Total output tensor size: {self.biomarker_config.total_output_size}")

    def _filter_age_records(self):
        """Filter out records with AGE = "90+" for HIPAA compliance (only if AGE is a biomarker being tested)"""
        
        # Check if AGE is actually a biomarker being tested
        age_is_biomarker = any(
            biomarker.name == 'AGE' 
            for biomarker in (self.biomarker_config.binary_biomarkers + 
                            self.biomarker_config.multiclass_biomarkers + 
                            self.biomarker_config.continuous_biomarkers)
        )
        
        if not age_is_biomarker:
            print("ℹ️  AGE is not a biomarker being tested - skipping age filtering")
            return self.df
        
        if 'AGE' not in self.df.columns:
            print("⚠️  AGE column not found - skipping age filtering")
            return self.df
        
        original_count = len(self.df)
        
        age_90_plus_mask = self.df['AGE'] == '90+'
        age_90_plus_count = age_90_plus_mask.sum()
        
        if age_90_plus_count > 0:
            print(f"🔒 HIPAA Compliance: Filtering out {age_90_plus_count:,} records with AGE='90+'")
            self.df = self.df[~age_90_plus_mask].copy()
        
        numeric_age_mask = pd.to_numeric(self.df['AGE'], errors='coerce').notna()
        if not numeric_age_mask.all():
            non_numeric_count = (~numeric_age_mask).sum()
            print(f"⚠️  Found {non_numeric_count} non-numeric AGE values, filtering them out")
            self.df = self.df[numeric_age_mask].copy()
        
        self.df['AGE'] = pd.to_numeric(self.df['AGE'], errors='coerce')
        
        if len(self.df) > 0:
            max_age = self.df['AGE'].max()
            min_age = self.df['AGE'].min()
            
            if max_age > 89:
                print(f"⚠️  Warning: Maximum age is {max_age}, expected <= 89")
            else:
                print(f"✅ Age range after filtering: {min_age:.0f} - {max_age:.0f} years")
        
        filtered_count = len(self.df)
        removed_count = original_count - filtered_count
        
        if removed_count > 0:
            print(f"📊 Dataset filtering summary:")
            print(f"   Original records: {original_count:,}")
            print(f"   Removed records: {removed_count:,}")
            print(f"   Remaining records: {filtered_count:,}")
            print(f"   Removal rate: {removed_count/original_count*100:.1f}%")
        
        return self.df

    def _prepare_all_targets(self):
        """Pre-compute all target tensors for the dataset"""
        import numpy as np
        
        targets = []
        for idx in range(len(self.df)):
            data = self.df.iloc[idx]
            
            # Create tensor with the configured size
            t = torch.zeros(self.biomarker_config.total_output_size, dtype=torch.float32)
            
            # Process binary biomarkers
            for biomarker in self.biomarker_config.binary_biomarkers:
                if biomarker.name in data:
                    layout = self.tensor_layout[biomarker.name]
                    idx_start = layout.start_idx
                    
                    # Convert using configured classes or default Condition enum
                    if biomarker.positive_class == "PRESENT" and biomarker.negative_class == "ABSENT":
                        # Use default Condition converter
                        from utils.labels import Condition
                        t[idx_start] = Condition.convert(data[biomarker.name])
                    else:
                        # Use custom class mapping
                        if data[biomarker.name] == biomarker.positive_class:
                            t[idx_start] = 1.0
                        elif data[biomarker.name] == biomarker.negative_class:
                            t[idx_start] = 0.0
                        else:
                            # Default to negative class for unknown values
                            t[idx_start] = 0.0
            
            # Process multiclass biomarkers
            for biomarker in self.biomarker_config.multiclass_biomarkers:
                if biomarker.name in data:
                    layout = self.tensor_layout[biomarker.name]
                    idx_start = layout.start_idx
                    idx_end = layout.end_idx
                    
                    # One-hot encoding
                    if data[biomarker.name] in biomarker.classes:
                        class_idx = biomarker.classes.index(data[biomarker.name])
                        t[idx_start + class_idx] = 1.0
            
            # Process continuous biomarkers
            for biomarker in self.biomarker_config.continuous_biomarkers:
                if biomarker.name in data:
                    layout = self.tensor_layout[biomarker.name]
                    idx_start = layout.start_idx
                    
                    # Normalize the continuous value
                    raw_value = float(data[biomarker.name])
                    normalized_value = biomarker.normalize(raw_value)
                    t[idx_start] = normalized_value
            
            targets.append(t.numpy())
        
        return np.array(targets)

    def __len__(self):
        """Get length of dataset"""
        return self.df.shape[0]

    def __getitem__(self, idx):
        """Get data at a certain index"""
        data = self.df.iloc[idx]
        
        # Get pre-computed targets
        t = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        # Load and process image
        xray = Image.open(os.path.join(self.data_path, 'data', data['FILE'] + '.png'))
        xray = xray.resize((self.size, self.size), Image.LANCZOS)
        xray = xray.convert('L')
        if self.transforms:
            xray = self.transforms(xray)
        
        return xray, t

    def at(self, idx):
        """Get directory name for a certain index"""
        return self.df.iloc[idx]['FILE'].split('.')[0]

    @property
    def filtering_summary(self):
        """Return a summary of filtering applied"""
        # Check if AGE is actually a biomarker being tested
        age_is_biomarker = any(
            biomarker.name == 'AGE' 
            for biomarker in (self.biomarker_config.binary_biomarkers + 
                            self.biomarker_config.multiclass_biomarkers + 
                            self.biomarker_config.continuous_biomarkers)
        )
        
        if age_is_biomarker:
            return "Age filtering applied (90+ records removed for HIPAA compliance)"
        else:
            return "No age filtering applied (AGE not being tested)"

def arg_parse():
    parser = ArgumentParser(description='Flexible Multi-Task Testing')
    parser.add_argument('--data_dir', required=True, help='Directory with test data')
    parser.add_argument('--checkpoint_path', required=True, help='Path to best_checkpoint.pth file')
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
    return parser.parse_args()

def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load checkpoint with the new format"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle both old and new checkpoint formats
    if 'model_state_dict' in checkpoint:
        # New format
        return checkpoint
    else:
        # Old format - convert
        return {
            'model_state_dict': checkpoint['state_dict'],
            'config': checkpoint.get('config', {}),
            'biomarker_config': checkpoint.get('biomarker_config', ''),
            'epoch': checkpoint.get('epoch', 0),
            'val_metrics': checkpoint.get('val_metrics', {})
        }

def create_model_from_checkpoint(checkpoint: Dict[str, Any], biomarker_config: FlexibleBiomarkerConfig) -> torch.nn.Module:
    """Create model architecture from checkpoint configuration"""
    
    # Get model configuration
    config_dict = checkpoint.get('config', {})
    if not config_dict:
        raise ValueError("No configuration found in checkpoint. Please use a checkpoint from the new training system.")
    
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
        image_augmentations=config_dict.get('image_augmentations', 'rotation (±15°), horizontal flip, random crop, color jitter (brightness±0.2, contrast±0.2), ImageNet normalization'),
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
    
    # Extract single_target_strategy from checkpoint
    single_target_strategy = config_dict.get('single_target_strategy', '')

    # Debug: Print checkpoint configuration
    print(f"🔍 Checkpoint config: {config_dict}")

    # Auto-detect strategy based on state_dict keys if needed
    state_dict_keys = list(checkpoint['model_state_dict'].keys())
    print(f"🔍 State dict keys (first 10): {state_dict_keys[:10]}")
    print(f"🔍 State dict keys with 'fc': {[k for k in state_dict_keys if 'fc' in k]}")

    has_feature_extractor = any('feature_extractor' in key for key in state_dict_keys)
    has_flattened_processor = any('flattened_processor' in key for key in state_dict_keys)
    has_feature_processor = any('feature_processor' in key for key in state_dict_keys)
    has_nested_resnet = any('resnet34.fc' in key for key in state_dict_keys)  # Custom ResNet34
    has_nested_resnet18 = any('resnet18.fc' in key for key in state_dict_keys)  # Custom ResNet18
    has_nested_resnet50 = any('resnet50.fc' in key for key in state_dict_keys)  # Custom ResNet50

    # Debug: Print detection results
    print(f"   Detection results: feature_extractor={has_feature_extractor}, flattened_processor={has_flattened_processor}, feature_processor={has_feature_processor}")
    print(f"   Nested ResNet detection: resnet34={has_nested_resnet}, resnet18={has_nested_resnet18}, resnet50={has_nested_resnet50}")

    detected_strategy = None
    if has_feature_extractor:
        if has_flattened_processor:
            # Model was trained with Direct classification head that adapted to flattened input
            detected_strategy = "Direct classification head"
        elif has_feature_processor and any('LayerNorm' in key or 'layer_norm' in key for key in state_dict_keys):
            # Model was trained with CLS token classification (uses LayerNorm)
            detected_strategy = "CLS token classification"
        elif has_feature_processor:
            # Model was trained with Direct classification head (uses BatchNorm)
            detected_strategy = "Direct classification head"
    elif has_nested_resnet or has_nested_resnet18 or has_nested_resnet50:
        # Custom ResNet with nested structure
        if has_nested_resnet:
            print(f"⚠️  Detected custom ResNet34 with nested structure (keys contain 'resnet34.fc')")
        elif has_nested_resnet18:
            print(f"⚠️  Detected custom ResNet18 with nested structure (keys contain 'resnet18.fc')")
        elif has_nested_resnet50:
            print(f"⚠️  Detected custom ResNet50 with nested structure (keys contain 'resnet50.fc')")
        detected_strategy = "Direct classification head"  # Most likely

    if detected_strategy and single_target_strategy != detected_strategy:
        print(f"⚠️  Auto-detected strategy mismatch: config says '{single_target_strategy}' but state_dict suggests '{detected_strategy}'")
        print(f"   State dict keys suggest: feature_extractor={has_feature_extractor}, flattened_processor={has_flattened_processor}, feature_processor={has_feature_processor}, nested_resnet34={has_nested_resnet}, nested_resnet18={has_nested_resnet18}, nested_resnet50={has_nested_resnet50}")
        single_target_strategy = detected_strategy
    
    print(f"Creating model: {config.model}")
    print(f"Fine-tuning strategy: {config.fine_tuning_strategy}")
    if single_target_strategy:
        print(f"Single-target strategy: {single_target_strategy}")
    
    # Create model using ModelFactory
    model = ModelFactory.create_model(
        architecture=config.model,
        num_classes=biomarker_config.total_output_size,
        pretrained_weights=config.pretrained_weights,
        fine_tuning_strategy=config.fine_tuning_strategy,
        dropout=config.dropout,
        biomarker_config=biomarker_config,
        single_target_strategy=single_target_strategy  # Use the extracted strategy
    )
    
    # Handle dynamic layer creation for DirectClassificationHeadExtractor
    print(f"   Debug: has_feature_processor={has_feature_processor}, has_flattened_processor={has_flattened_processor}")
    print(f"   Debug: hasattr(model, 'fc')={hasattr(model, 'fc') if 'model' in locals() else 'model not created yet'}")

    # Check if model has fc attribute (standard ResNet) or classifier attribute (custom models)
    has_fc = hasattr(model, 'fc') if 'model' in locals() else False
    has_classifier = hasattr(model, 'classifier') if 'model' in locals() else False

    print(f"   Debug: model has fc={has_fc}, classifier={has_classifier}")

    if has_feature_processor and has_flattened_processor and (has_fc or has_classifier):
        print("🔧 Pre-creating both feature_processor and flattened_processor to match saved state dict...")
        print("   ✅ Dual processor creation logic triggered!")

        # Find feature_processor keys to understand its structure
        feature_processor_keys = [k for k in state_dict_keys if 'feature_processor' in k and '.weight' in k]
        flattened_processor_keys = [k for k in state_dict_keys if 'flattened_processor' in k and '.weight' in k]

        print(f"   Found feature_processor keys: {[k for k in feature_processor_keys]}")
        print(f"   Found flattened_processor keys: {[k for k in flattened_processor_keys]}")

        if feature_processor_keys and flattened_processor_keys:
            import torch.nn as nn

            # Extract dimensions from the first layer of each processor
            # Feature processor layer 1 (Linear)
            fp_layer1_key = [k for k in feature_processor_keys if '.1.weight' in k][0]
            fp_weight_tensor = checkpoint['model_state_dict'][fp_layer1_key]
            fp_input_dim = fp_weight_tensor.shape[1]
            fp_output_dim = fp_weight_tensor.shape[0]

            # Flattened processor layer 0 (Linear)
            flp_layer0_key = [k for k in flattened_processor_keys if '.0.weight' in k][0]
            flp_weight_tensor = checkpoint['model_state_dict'][flp_layer0_key]
            flp_input_dim = flp_weight_tensor.shape[1]
            flp_output_dim = flp_weight_tensor.shape[0]

            print(f"   Feature processor: {fp_input_dim} -> {fp_output_dim}")
            print(f"   Flattened processor: {flp_input_dim} -> {flp_output_dim}")

            # Create the processors on the correct attribute (fc or classifier)
            if has_fc:
                # Standard ResNet has fc attribute
                target_extractor = model.fc.feature_extractor
            else:
                # Custom models have classifier attribute
                target_extractor = model.classifier.feature_extractor

            # Create the feature processor (Linear -> ReLU -> Dropout -> BatchNorm)
            target_extractor.feature_processor = nn.Sequential(
                nn.Linear(fp_input_dim, fp_output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.BatchNorm1d(fp_output_dim)
            ).to(device)

            # Create the flattened processor (Linear -> ReLU -> Dropout -> LayerNorm)
            target_extractor.flattened_processor = nn.Sequential(
                nn.Linear(flp_input_dim, flp_output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.LayerNorm(flp_output_dim)
            ).to(device)

            print("   Created both processors with correct dimensions")

            # Now remap all the keys to match the dynamic layers
            print("🔧 Remapping processor keys...")
            state_dict = checkpoint['model_state_dict'].copy()
            remapped_state_dict = {}

            # Determine the correct prefix based on model structure
            if has_fc:
                extractor_prefix = "fc.feature_extractor"
            else:
                extractor_prefix = "classifier.feature_extractor"

            # Map feature_processor keys
            for old_key, value in state_dict.items():
                if 'feature_processor' in old_key:
                    # Extract layer index and parameter name
                    parts = old_key.split('.')
                    if len(parts) >= 3:
                        layer_idx = parts[-2]  # '1', '4', etc.
                        param_name = parts[-1]  # 'weight', 'bias', etc.
                        new_key = f"{extractor_prefix}.feature_processor.{layer_idx}.{param_name}"
                        remapped_state_dict[new_key] = value
                        print(f"   Remapped feature: {old_key} -> {new_key}")
                    else:
                        remapped_state_dict[old_key] = value
                elif 'flattened_processor' in old_key:
                    # Extract layer index and parameter name
                    parts = old_key.split('.')
                    if len(parts) >= 3:
                        layer_idx = parts[-2]  # '0', '3', etc.
                        param_name = parts[-1]  # 'weight', 'bias'
                        new_key = f"{extractor_prefix}.flattened_processor.{layer_idx}.{param_name}"
                        remapped_state_dict[new_key] = value
                        print(f"   Remapped flattened: {old_key} -> {new_key}")
                    else:
                        remapped_state_dict[old_key] = value
                else:
                    remapped_state_dict[old_key] = value

            checkpoint = checkpoint.copy()
            checkpoint['model_state_dict'] = remapped_state_dict

    # Handle custom ResNet nested structures
    if (has_nested_resnet and config.model == "ResNet-34") or (has_nested_resnet18 and config.model == "ResNet-18") or (has_nested_resnet50 and config.model == "ResNet-50"):
        if has_nested_resnet and config.model == "ResNet-34":
            nested_type = "ResNet-34"
            old_prefix = "resnet34.fc."
            new_prefix = "fc."
        elif has_nested_resnet18 and config.model == "ResNet-18":
            nested_type = "ResNet-18"
            old_prefix = "resnet18.fc."
            new_prefix = "fc."
        elif has_nested_resnet50 and config.model == "ResNet-50":
            nested_type = "ResNet-50"
            old_prefix = "resnet50.fc."
            new_prefix = "fc."
        else:
            nested_type = "Unknown"
            old_prefix = ""
            new_prefix = ""

        print(f"🔧 Handling custom {nested_type} nested structure...")
        # The saved model has keys like 'resnetXX.fc.weight' but we expect 'fc.weight'
        # We'll need to remap these keys during loading
        state_dict = checkpoint['model_state_dict'].copy()

        # Remap nested keys to standard keys
        remapped_state_dict = {}
        for old_key, value in state_dict.items():
            if old_prefix in old_key:
                new_key = old_key.replace(old_prefix, new_prefix)
                remapped_state_dict[new_key] = value
                print(f"   Remapped: {old_key} -> {new_key}")
            else:
                remapped_state_dict[old_key] = value

        checkpoint = checkpoint.copy()
        checkpoint['model_state_dict'] = remapped_state_dict

    # Try to load the state dict with strict=False to handle mismatches gracefully
    try:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing_keys or unexpected_keys:
            print(f"⚠️  State dict loading warnings:")
            if missing_keys:
                print(f"  Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"  Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            print("  ✅ Model loaded successfully despite key mismatches")
        else:
            print("✅ Model state dict loaded perfectly!")
    except Exception as e:
        print(f"❌ Failed to load state dict even with strict=False: {e}")
        raise e
    model.to(device)
    model.eval()
    
    # Log model info
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
    
    print(f"🔍 Test preprocessing settings:")
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
        print(f"   ✅ Normalization applied: {config.pretrained_weights} normalization")
    else:
        print(f"   ⚠️  No normalization applied (imagenet_norm=False)")
    
    return transforms.Compose(transform_list)

def create_test_dataset(data_dir: str, biomarker_config: FlexibleBiomarkerConfig, 
                       config: ExperimentConfig, size: int = 256, only_pred: bool = False, 
                       test_csv: str = 'test.csv', batch_size: int = 16) -> DataLoader:
    """Create test dataset and dataloader with matching preprocessing"""
    
    # Create transforms that match training exactly
    transform = create_test_transforms(config)
    
    if only_pred:
        # Test dataset without labels
        test_dataset = TestDataset(data_dir, transforms=transform, size=size)
        print(f"Created test dataset with {len(test_dataset)} images (prediction only)")
    else:
        # Create a custom dataset that can handle any CSV file
        test_dataset = CustomCSVDataset(
            data_dir, 
            biomarker_config, 
            transforms=transform, 
            size=size, 
            csv_file=test_csv
        )
        print(f"Created test dataset with {len(test_dataset)} samples")
        print(f"Dataset filtering applied: {test_dataset.filtering_summary}")
    
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
        
        # Convert targets to tensor format
        target_tensors = []
        for study_id in study_ids:
            if study_id in all_targets:
                target_tensors.append(torch.from_numpy(all_targets[study_id]))
        
        if target_tensors:
            target_tensor = torch.stack(target_tensors).to(device)
            all_predictions_gpu = all_predictions.to(device)
            metrics = calculate_enhanced_metrics(all_predictions_gpu, target_tensor, biomarker_config, optimal_thresholds)
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
    checkpoint = load_checkpoint(args.checkpoint_path)
    
    # Create model and get config
    model, config = create_model_from_checkpoint(checkpoint, biomarker_config)
    
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
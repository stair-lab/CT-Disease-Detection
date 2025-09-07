#!/usr/bin/env python3
"""
Test script to verify the enhanced training setup
"""

import os
import sys
import torch
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from model.model_factory import ModelFactory, MultiTaskHead
        print("✅ Model factory imports successful")
    except ImportError as e:
        print(f"❌ Model factory import failed: {e}")
        return False
    
    try:
        from config.experiment_config import ExperimentConfigLoader, ExperimentConfig
        print("✅ Config system imports successful")
    except ImportError as e:
        print(f"❌ Config system import failed: {e}")
        return False
    
    try:
        import timm
        print("✅ TIMM import successful")
    except ImportError:
        print("⚠️  TIMM not available - some models will use placeholders")
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("✅ TensorBoard import successful")
    except ImportError:
        print("❌ TensorBoard not available - logging will fail")
        return False
    
    return True


def test_model_factory():
    """Test model factory functionality"""
    print("\nTesting model factory...")
    
    try:
        from model.model_factory import ModelFactory, MultiTaskHead
        
        # Test ResNet-18 creation
        model = ModelFactory.create_model("ResNet-18", num_classes=13)
        print(f"✅ Created ResNet-18 with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        x = torch.randn(2, 1, 256, 256)
        with torch.no_grad():
            output = model(x)
        print(f"✅ Forward pass successful, output shape: {output.shape}")
        
        # Test multi-task head
        head = MultiTaskHead(512, num_binary_tasks=7, num_calcium_classes=4, num_regression_tasks=2)
        x_head = torch.randn(2, 512)
        with torch.no_grad():
            output_head = head(x_head)
        print(f"✅ Multi-task head successful, output shape: {output_head.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model factory test failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    # Create a minimal test CSV
    test_csv_content = """Model,Loss Function,Must Include,Learning Rate,Batch Size,Weight Decay,Optimizer,Scheduler,Image Augmentations,Dropout,Loss-Specific Params,Multi_Target_Strategy,Pretrained_Weights,Fine_Tuning_Strategy,Expected_GPU_Memory,Architectural_Family,Class_Weighting,Sampling_Strategy,Threshold_Selection
ResNet-18,CE,Yes,"[1e-4, 1e-3]",16,1e-5,AdamW,CosineAnnealing,"rotation (±15°), horizontal flip",0.1,"class_weights=inverse_frequency",Shared backbone + task-specific heads,ImageNet,Full fine-tuning,4-6GB,CNN,inverse_frequency,balanced_batch,F1_optimal"""
    
    test_csv_path = "test_config.csv"
    with open(test_csv_path, 'w') as f:
        f.write(test_csv_content)
    
    try:
        from config.experiment_config import ExperimentConfigLoader
        loader = ExperimentConfigLoader(test_csv_path)
        configs = loader.load_all_configs()
        
        if len(configs) == 1:
            config = configs[0]
            print(f"✅ Loaded config for {config.model}")
            print(f"✅ Learning rates: {config.learning_rate}")
            print(f"✅ Experiment name: {config.experiment_name}")
            
            # Clean up
            os.remove(test_csv_path)
            return True
        else:
            print(f"❌ Expected 1 config, got {len(configs)}")
            os.remove(test_csv_path)
            return False
            
    except Exception as e:
        print(f"❌ Config loading test failed: {e}")
        if os.path.exists(test_csv_path):
            os.remove(test_csv_path)
        return False


def test_data_transforms():
    """Test data transformation pipeline"""
    print("\nTesting data transforms...")
    
    try:
        from config.experiment_config import parse_augmentation_string
        from train_enhanced import create_data_transforms, ExperimentConfig
        
        # Create a test config
        config = ExperimentConfig(
            model="ResNet-18",
            loss_function="CE",
            must_include=True,
            learning_rate=[1e-4],
            batch_size=16,
            weight_decay=1e-5,
            optimizer="AdamW",
            scheduler="CosineAnnealing",
            image_augmentations="rotation (±15°), horizontal flip, random crop, color jitter (brightness±0.2, contrast±0.2), ImageNet normalization",
            dropout=0.1,
            loss_specific_params="class_weights=inverse_frequency",
            multi_target_strategy="Shared backbone + task-specific heads",
            pretrained_weights="ImageNet",
            fine_tuning_strategy="Full fine-tuning",
            expected_gpu_memory="4-6GB",
            architectural_family="CNN",
            class_weighting="inverse_frequency",
            sampling_strategy="balanced_batch",
            threshold_selection="F1_optimal"
        )
        
        # Test augmentation parsing
        aug_params = parse_augmentation_string(config.image_augmentations)
        print(f"✅ Parsed augmentations: {aug_params}")
        
        # Test transform creation
        train_transform = create_data_transforms(config, is_training=True)
        val_transform = create_data_transforms(config, is_training=False)
        print("✅ Created data transforms")
        
        return True
        
    except Exception as e:
        print(f"❌ Data transforms test failed: {e}")
        return False


def check_gpu_setup():
    """Check GPU availability and memory"""
    print("\nChecking GPU setup...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA available with {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return True
    else:
        print("⚠️  CUDA not available - will run on CPU (very slow)")
        return False


def main():
    """Run all tests"""
    print("🧪 Testing Enhanced Multi-Task Training Setup")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_factory,
        test_config_loading,
        test_data_transforms,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Check GPU (informational only)
    check_gpu_setup()
    
    # Summary
    print("\n" + "=" * 50)
    print("🧪 Test Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed! Setup is ready.")
        return True
    else:
        print(f"❌ {passed}/{total} tests passed. Please fix the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Example usage of GradNorm for multi-task loss balancing
"""

import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def example_1_command_line_usage():
    """
    Example 1: Using GradNorm via command line arguments
    """
    print("=== Example 1: Command Line Usage ===")
    print()
    
    print("To train a single model with GradNorm:")
    print("python train.py \\")
    print("    --config_csv experimentation_plan_simplified.csv \\")
    print("    --data_dir /path/to/your/data \\")
    print("    --biomarker_config config/biomarker_config_default.yaml \\")
    print("    --model_name 'ResNet50' \\")
    print("    --use_gradnorm \\")
    print("    --gradnorm_alpha 0.16 \\")
    print("    --gradnorm_update_freq 10")
    print()
    
    print("To run experiments with GradNorm via run_experiments.py:")
    print("python run_experiments.py \\")
    print("    --data_dir /path/to/your/data \\")
    print("    --biomarker_config config/biomarker_config_default.yaml \\")
    print("    --must_include_only \\")
    print("    --use_gradnorm \\")
    print("    --gradnorm_alpha 0.16 \\")
    print("    --gradnorm_update_freq 10")
    print()


def example_2_csv_configuration():
    """
    Example 2: Adding GradNorm columns to your CSV configuration
    """
    print("=== Example 2: CSV Configuration ===")
    print()
    
    print("Add these columns to your experimentation_plan_simplified.csv:")
    print("- Use_GradNorm: 'Yes' or 'No'")
    print("- GradNorm_Alpha: float (default: 0.16)")
    print("- GradNorm_Update_Freq: int (default: 10)")
    print()
    
    print("Example CSV row:")
    print("Model,Loss Function,...,Use_GradNorm,GradNorm_Alpha,GradNorm_Update_Freq")
    print("ResNet50,MultiTaskLoss,...,Yes,0.16,10")
    print("DenseNet121,MultiTaskLoss,...,No,0.16,10")
    print()


def example_3_programmatic_usage():
    """
    Example 3: Using GradNorm programmatically
    """
    print("=== Example 3: Programmatic Usage ===")
    print()
    
    code = '''
import torch
from config.biomarker_config import FlexibleBiomarkerConfig
from model.gradnorm_loss import GradNormLoss, GradNormTrainer

# Load your biomarker configuration
biomarker_config = FlexibleBiomarkerConfig('config/biomarker_config_default.yaml')

# Create GradNorm loss
gradnorm_loss = GradNormLoss(
    biomarker_config=biomarker_config,
    alpha=0.16,  # Restoring force strength
    update_weights_every=10,  # Update frequency
    initial_task_loss_average_window=20,  # Window for initial loss averaging
    normalize_losses=True,  # Normalize individual losses
    restoring_force_factor=0.1  # Restoring force factor
)

# Create trainer helper
gradnorm_trainer = GradNormTrainer(gradnorm_loss)

# In your training loop:
for batch in dataloader:
    images, targets = batch
    predictions = model(images)
    
    # Use GradNorm for loss computation
    loss, loss_dict = gradnorm_trainer.compute_loss(model, predictions, targets)
    
    # Standard backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Get current task weights for logging
    task_weights = gradnorm_trainer.get_training_stats()['task_weights']
    print(f"Current task weights: {task_weights}")
'''
    
    print(code)
    print()


def example_4_monitoring_gradnorm():
    """
    Example 4: Monitoring GradNorm during training
    """
    print("=== Example 4: Monitoring GradNorm ===")
    print()
    
    print("GradNorm provides several ways to monitor training:")
    print()
    
    print("1. TensorBoard Logs:")
    print("   - GradNorm_Weights/{task_name}: Current weight for each task")
    print("   - Look in the tensorboard logs directory")
    print()
    
    print("2. Console Output:")
    print("   - Initial task loss averages are printed when computed")
    print("   - Task weights are logged every epoch if GradNorm is enabled")
    print("   - Weight updates are printed every 50 updates")
    print()
    
    print("3. Programmatic Access:")
    code = '''
# Get current training statistics
stats = gradnorm_trainer.get_training_stats()
print(f"Task weights: {stats['task_weights']}")
print(f"Step count: {stats['step_count']}")
print(f"Initial losses computed: {stats['initial_losses_computed']}")

# Get weight history for analysis
weight_history = gradnorm_loss.get_weight_history()
for i, weights in enumerate(weight_history[-5:]):  # Last 5 updates
    print(f"Update {i}: {weights}")
'''
    print(code)
    print()


def example_5_hyperparameter_tuning():
    """
    Example 5: GradNorm hyperparameter guidelines
    """
    print("=== Example 5: Hyperparameter Guidelines ===")
    print()
    
    print("Key GradNorm hyperparameters:")
    print()
    
    print("1. alpha (restoring force strength):")
    print("   - Typical range: 0.12 - 0.16")
    print("   - Lower values: More aggressive balancing")
    print("   - Higher values: More conservative balancing")
    print("   - Start with 0.16, reduce if tasks are very imbalanced")
    print()
    
    print("2. update_weights_every:")
    print("   - Typical range: 5 - 20 iterations")
    print("   - More frequent updates: Faster adaptation, more overhead")
    print("   - Less frequent updates: Smoother adaptation, less overhead")
    print("   - Start with 10")
    print()
    
    print("3. initial_task_loss_average_window:")
    print("   - Typical range: 20 - 50 iterations")
    print("   - Larger window: More stable initial averages")
    print("   - Smaller window: Faster initialization")
    print("   - Start with 20")
    print()
    
    print("4. restoring_force_factor:")
    print("   - Typical range: 0.05 - 0.2")
    print("   - Controls the strength of weight updates")
    print("   - Start with 0.1")
    print()


def main():
    """Run all examples"""
    print("GradNorm Multi-Task Loss Balancing - Usage Examples")
    print("=" * 60)
    print()
    
    example_1_command_line_usage()
    example_2_csv_configuration()
    example_3_programmatic_usage()
    example_4_monitoring_gradnorm()
    example_5_hyperparameter_tuning()
    
    print("=== Key Benefits of GradNorm ===")
    print()
    print("✅ Automatic loss balancing - no manual tuning needed")
    print("✅ Adapts to task difficulty changes during training")
    print("✅ Prevents dominant tasks from overwhelming others")
    print("✅ Improves convergence for multi-task learning")
    print("✅ Extensive logging and monitoring capabilities")
    print()
    
    print("=== When to Use GradNorm ===")
    print()
    print("🎯 Multi-task learning with imbalanced tasks")
    print("🎯 Tasks with very different loss scales")
    print("🎯 When some tasks converge much faster than others")
    print("🎯 Complex biomarker prediction with mixed task types")
    print()


if __name__ == "__main__":
    main()

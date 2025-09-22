# GradNorm Multi-Task Loss Balancing Guide

This guide explains how to use GradNorm for automatic loss balancing in your multi-task comorbidity detection experiments.

## Overview

GradNorm is an algorithm that automatically balances losses in multi-task learning by adjusting task weights based on gradient magnitudes. This is particularly useful when:

- Tasks have very different loss scales
- Some tasks converge much faster than others
- You want to prevent dominant tasks from overwhelming smaller ones
- Manual loss weight tuning is impractical

## Key Features

✅ **Automatic Loss Balancing**: No manual tuning of task weights needed  
✅ **Adaptive**: Adjusts to changing task difficulties during training  
✅ **Comprehensive Logging**: Full TensorBoard and console logging  
✅ **Flexible Configuration**: CSV, command-line, and programmatic configuration  
✅ **Production Ready**: Integrated with existing training pipeline  

## Quick Start

### 1. Command Line Usage

Train a single model with GradNorm:

```bash
python train.py \
    --config_csv experimentation_plan_simplified.csv \
    --data_dir /path/to/your/data \
    --biomarker_config config/biomarker_config_default.yaml \
    --model_name 'ResNet50' \
    --use_gradnorm \
    --gradnorm_alpha 0.16 \
    --gradnorm_update_freq 10
```

Run multiple experiments with GradNorm:

```bash
python run_experiments.py \
    --data_dir /path/to/your/data \
    --biomarker_config config/biomarker_config_default.yaml \
    --must_include_only \
    --use_gradnorm \
    --gradnorm_alpha 0.16 \
    --gradnorm_update_freq 10
```

### 2. CSV Configuration

Add these columns to your `experimentation_plan_simplified.csv`:

| Column | Description | Default |
|--------|-------------|---------|
| `Use_GradNorm` | Enable GradNorm (`Yes`/`No`) | `No` |
| `GradNorm_Alpha` | Restoring force strength | `0.16` |
| `GradNorm_Update_Freq` | Update frequency (iterations) | `10` |

Example CSV row:
```csv
Model,Loss Function,...,Use_GradNorm,GradNorm_Alpha,GradNorm_Update_Freq
ResNet50,MultiTaskLoss,...,Yes,0.16,10
DenseNet121,MultiTaskLoss,...,No,0.16,10
```

## Algorithm Details

### How GradNorm Works

1. **Initial Phase**: Collects initial task losses over a window (default: 20 iterations)
2. **Weight Computation**: Calculates gradient norms for each task on shared parameters
3. **Target Setting**: Sets target gradient norms based on relative training rates
4. **Weight Updates**: Updates task weights to minimize difference between actual and target gradients
5. **Loss Balancing**: Applies weights to individual task losses

### Mathematical Foundation

GradNorm minimizes the loss:

```
L_GradNorm = Σ |G_i - Ḡ × (r_i)^α|
```

Where:
- `G_i`: Gradient norm for task i
- `Ḡ`: Average gradient norm across tasks
- `r_i`: Relative inverse training rate for task i
- `α`: Restoring force strength (hyperparameter)

## Configuration Parameters

### Core Parameters

#### `alpha` (Restoring Force Strength)
- **Range**: 0.12 - 0.16
- **Default**: 0.16
- **Effect**: Controls how aggressively GradNorm balances tasks
- **Lower values**: More aggressive balancing
- **Higher values**: More conservative balancing

#### `update_weights_every` (Update Frequency)
- **Range**: 5 - 20 iterations
- **Default**: 10
- **Effect**: How often task weights are updated
- **More frequent**: Faster adaptation, more computational overhead
- **Less frequent**: Smoother adaptation, less overhead

### Advanced Parameters

#### `initial_task_loss_average_window`
- **Range**: 20 - 50 iterations
- **Default**: 20
- **Effect**: Window size for computing initial task loss averages
- **Larger**: More stable initial estimates
- **Smaller**: Faster initialization

#### `normalize_losses`
- **Default**: `True`
- **Effect**: Whether to normalize individual task losses by their initial averages
- **Recommended**: Keep as `True` for most cases

#### `restoring_force_factor`
- **Range**: 0.05 - 0.2
- **Default**: 0.1
- **Effect**: Controls the magnitude of weight updates
- **Higher**: More aggressive weight changes
- **Lower**: More conservative weight changes

## Monitoring and Debugging

### TensorBoard Logs

GradNorm automatically logs to TensorBoard:

```bash
tensorboard --logdir /path/to/your/output/directory
```

**Available metrics:**
- `GradNorm_Weights/{task_name}`: Current weight for each task
- `Loss/Train`: Overall training loss
- `Loss/Validation`: Overall validation loss

### Console Output

**Initial phase:**
```
GradNorm: Initial task loss averages computed: {'binary_hypertension': 0.693, 'binary_diabetes': 0.421, ...}
```

**Weight updates:**
```
GradNorm Step 100: Weights = {'binary_hypertension': 1.2, 'binary_diabetes': 0.8, ...}
```

**Per-epoch summaries:**
```
GradNorm task weights: {'binary_hypertension': 1.15, 'binary_diabetes': 0.85, ...}
```

### Programmatic Monitoring

```python
# Get current statistics
stats = gradnorm_trainer.get_training_stats()
print(f"Task weights: {stats['task_weights']}")
print(f"Step count: {stats['step_count']}")

# Get weight evolution history
weight_history = gradnorm_loss.get_weight_history()
```

## Best Practices

### 1. Hyperparameter Selection

**Start with defaults:**
- `alpha = 0.16`
- `update_weights_every = 10`
- `initial_task_loss_average_window = 20`

**Adjust based on behavior:**
- If tasks remain imbalanced: Reduce `alpha` to 0.12
- If weights oscillate too much: Increase `update_weights_every`
- If initialization is unstable: Increase `initial_task_loss_average_window`

### 2. Monitoring Strategy

1. **Watch initial averages**: Ensure they're reasonable for your tasks
2. **Monitor weight evolution**: Weights should stabilize after initial adaptation
3. **Check individual task performance**: Ensure no task is being completely suppressed
4. **Compare with baseline**: Run experiments with and without GradNorm

### 3. Common Issues and Solutions

**Issue**: Weights oscillate wildly
- **Solution**: Increase `update_weights_every` or reduce `restoring_force_factor`

**Issue**: One task dominates completely
- **Solution**: Reduce `alpha` or check for bugs in task definitions

**Issue**: Training is slower
- **Solution**: Increase `update_weights_every` to reduce computational overhead

**Issue**: No improvement over baseline
- **Solution**: Your tasks may already be well-balanced, or try different `alpha` values

## Integration with Existing Code

### Minimal Integration

If you have existing training code, minimal changes are needed:

```python
# Replace this:
# loss, loss_dict = criterion(predictions, targets)

# With this:
if gradnorm_trainer is not None:
    loss, loss_dict = gradnorm_trainer.compute_loss(model, predictions, targets)
else:
    loss, loss_dict = criterion(predictions, targets)
```

### Full Integration

The codebase provides complete integration:
- Automatic configuration loading from CSV
- Command-line argument support
- TensorBoard logging
- Checkpoint saving/loading compatibility

## Performance Considerations

### Computational Overhead

GradNorm adds minimal overhead:
- **Memory**: ~1KB per task for weight parameters
- **Computation**: Gradient computation every `update_weights_every` iterations
- **Typical overhead**: <5% of total training time

### Scaling

GradNorm scales well with:
- **Number of tasks**: Linear scaling
- **Model size**: Overhead is independent of model size
- **Batch size**: No additional scaling

## Troubleshooting

### Common Error Messages

**"Could not find shared parameters for GradNorm"**
- **Cause**: Model architecture not recognized
- **Solution**: Ensure your model has parameters not in task-specific heads

**"Initial task loss averages not computed"**
- **Cause**: Not enough training iterations
- **Solution**: Reduce `initial_task_loss_average_window` or train longer

### Debugging Tips

1. **Enable verbose logging**: Check console output for GradNorm messages
2. **Monitor TensorBoard**: Watch weight evolution graphs
3. **Compare with baseline**: Always compare against non-GradNorm training
4. **Check task definitions**: Ensure all tasks are properly defined in biomarker config

## Examples and Use Cases

### Example 1: Imbalanced Medical Tasks

```yaml
# biomarker_config.yaml with very different task difficulties
binary_biomarkers:
  - name: "rare_disease"      # Very few positive cases
    weight: 1.0
  - name: "common_condition"  # Many positive cases  
    weight: 1.0
```

Without GradNorm: `common_condition` dominates training  
With GradNorm: Both tasks receive appropriate attention

### Example 2: Mixed Task Types

```yaml
binary_biomarkers:
  - name: "hypertension"
multiclass_biomarkers:
  - name: "severity_score"
    num_classes: 5
continuous_biomarkers:
  - name: "blood_pressure"
```

GradNorm automatically balances between classification and regression losses.

## References

- **Original Paper**: "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks" (Chen et al., 2018)
- **ArXiv**: https://arxiv.org/abs/1711.02257
- **Implementation**: Based on the official algorithm with adaptations for medical multi-task learning

## Support

For issues or questions:
1. Check the examples in `examples/gradnorm_usage.py`
2. Review TensorBoard logs for debugging
3. Compare with baseline experiments
4. Adjust hyperparameters based on your specific use case

---

*This implementation of GradNorm has been specifically adapted for the Comorbidities-Detection project's flexible biomarker configuration system.*

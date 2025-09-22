#!/bin/bash

# Examples of using launch_parallel_experiments.sh with GradNorm
# This file demonstrates different ways to launch parallel experiments with GradNorm

echo "🚀 GradNorm Parallel Experiment Launch Examples"
echo "=============================================="
echo ""

# Make the script executable
chmod +x ../launch_parallel_experiments.sh

echo "📋 Example 1: Launch experiments WITHOUT GradNorm (default behavior)"
echo "Command:"
echo "  ./launch_parallel_experiments.sh \"ResNet-18\" \"EfficientNet-B0\""
echo ""

echo "📋 Example 2: Launch experiments WITH GradNorm (default settings)"
echo "Command:"
echo "  ./launch_parallel_experiments.sh --use-gradnorm \"ResNet-18\" \"EfficientNet-B0\""
echo ""

echo "📋 Example 3: Launch experiments with custom GradNorm settings"
echo "Command:"
echo "  ./launch_parallel_experiments.sh --use-gradnorm --gradnorm-alpha 0.12 --gradnorm-update-freq 5 \"ResNet-18\""
echo ""

echo "📋 Example 4: Launch multiple models with GradNorm"
echo "Command:"
echo "  ./launch_parallel_experiments.sh --use-gradnorm \"ResNet-18\" \"ResNet-34\" \"EfficientNet-B0\" \"DenseNet-121\""
echo ""

echo "📋 Example 5: Get help information"
echo "Command:"
echo "  ./launch_parallel_experiments.sh --help"
echo ""

echo "🔧 GradNorm Parameter Guidelines:"
echo "  --gradnorm-alpha: 0.12-0.16 (default: 0.16)"
echo "    - Lower values = more aggressive balancing"
echo "    - Higher values = more conservative balancing"
echo ""
echo "  --gradnorm-update-freq: 5-20 (default: 10)"
echo "    - Lower values = more frequent updates, more overhead"
echo "    - Higher values = less frequent updates, less overhead"
echo ""

echo "📊 What happens when you run with GradNorm:"
echo "  ✅ Automatic loss balancing across all tasks"
echo "  ✅ Task weights logged to TensorBoard"
echo "  ✅ Console output shows GradNorm progress"
echo "  ✅ Session names include 'gradnorm' identifier"
echo "  ✅ All existing functionality preserved"
echo ""

echo "🔍 Monitoring GradNorm experiments:"
echo "  # List all GradNorm sessions"
echo "  tmux list-sessions | grep gradnorm"
echo ""
echo "  # Attach to a specific GradNorm experiment"
echo "  tmux attach -t ResNet_18_mlhc_targets_linear_probe_regularized_gradnorm_lr_1e-4_gpu_1"
echo ""
echo "  # Stop all GradNorm experiments"
echo "  tmux list-sessions | grep gradnorm | cut -d: -f1 | xargs -I {} tmux kill-session -t {}"
echo ""

echo "📈 TensorBoard metrics to watch:"
echo "  - GradNorm_Weights/{task_name}: Current weight for each task"
echo "  - Metrics/Median_AUROC_Val: Validation median AUROC (used for checkpoint selection)"
echo "  - Metrics/Average_AUROC_Val: Validation average AUROC (for comparison)"
echo ""

echo "💡 Tips:"
echo "  - Start with default GradNorm settings (alpha=0.16, update_freq=10)"
echo "  - Monitor task weights in TensorBoard to see balancing in action"
echo "  - Compare median vs average AUROC to detect outlier tasks"
echo "  - Use --resume flag to continue interrupted experiments"
echo ""

echo "🎯 Ready to launch! Choose an example above and run it."

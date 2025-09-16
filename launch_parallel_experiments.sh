#!/bin/bash

# Parallel EfficientNet-B0 Regularized Launcher
# Launches experiments on multiple GPUs simultaneously with updated regularization parameters

echo "🚀 LAUNCHING PARALLEL ResNet-34 MLHC Targets EXPERIMENTS"
echo "=============================================================="
echo "🎯 Model: ResNet-34 (HCC18 Only)"
echo "🔧 Regularization: Dropout 0.1, Weight Decay 1e-05"
echo "💾 Memory requirement: 6-8GB per experiment"
echo "🖥️ Available GPUs: 3 (GPUs 0, 1, 2)"
echo "📊 Learning rates: [1e-5, 1e-4, 1e-3] = 3 experiments"
echo ""

# Configuration
PYTHON_PATH="/lfs/turing1/0/mahmedc/miniconda3/envs/mahmedc_env/bin/python"
MODEL_NAME="ResNet-34"
AVAILABLE_GPUS=(0 1 2)
LEARNING_RATES=(1e-5 1e-4 1e-3)

# Now we can run true parallel experiments with specific learning rates on different GPUs!
echo "🏃 Starting parallel ResNet-34 HCC18 Only experiments..."

# Base command template
BASE_CMD="$PYTHON_PATH run_experiments.py \
    --config_csv experimentation_plan_simplified.csv \
    --biomarker_config config/biomarker_config_hcc18.yaml \
    --data_dir /lfs/turing1/0/mahmedc/Comorbidities-Detection/datasets/full_data \
    --model_name \"$MODEL_NAME\" \
    --output_base_dir /lfs/turing1/0/mahmedc/Comorbidities-Detection/models/hcc18_only \
    --resume \
    --no_confirm \
    --epochs 100"

# Launch experiments for each learning rate on different GPUs
for i in "${!LEARNING_RATES[@]}"; do
    LR="${LEARNING_RATES[$i]}"
    GPU="${AVAILABLE_GPUS[$i]}"
    SESSION_NAME="resnet_34_hcc18_only_lr_${LR}_gpu_${GPU}"
    
    echo "  📋 GPU $GPU: Learning rate $LR (tmux: $SESSION_NAME)"
    
    # Create tmux session and run experiment
    tmux new-session -d -s "$SESSION_NAME" -c "/lfs/turing1/0/mahmedc/Comorbidities-Detection/CT-Disease-Detection"
    
    # Send commands to tmux session
    tmux send-keys -t "$SESSION_NAME" "export CUDA_VISIBLE_DEVICES=$GPU" Enter
    tmux send-keys -t "$SESSION_NAME" "export HF_HOME=/lfs/turing1/0/mahmedc/.cache/huggingface" Enter
    tmux send-keys -t "$SESSION_NAME" "export HF_HUB_CACHE=/lfs/turing1/0/mahmedc/.cache/huggingface/hub" Enter
    tmux send-keys -t "$SESSION_NAME" "echo \"🚀 Starting ResNet-34 HCC18 Only - LR: $LR, GPU: $GPU\"" Enter
    tmux send-keys -t "$SESSION_NAME" "$BASE_CMD --learning_rate $LR" Enter
    
    sleep 2  # Brief delay between launches
done

echo ""
echo "✅ All experiments launched!"
echo ""
echo "📋 Monitor experiments:"
echo "  tmux list-sessions | grep resnet_34_hcc18_only"
echo ""
echo "🔍 Attach to specific experiment:"
for i in "${!LEARNING_RATES[@]}"; do
    LR="${LEARNING_RATES[$i]}"
    GPU="${AVAILABLE_GPUS[$i]}"
    SESSION_NAME="resnet_34_hcc18_only_lr_${LR}_gpu_${GPU}"
    echo "  tmux attach -t $SESSION_NAME  # LR: $LR, GPU: $GPU"
done
echo ""
echo "📊 Check GPU usage:"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "📁 Results will be saved in: /lfs/turing1/0/mahmedc/Comorbidities-Detection/models/hcc18_only"

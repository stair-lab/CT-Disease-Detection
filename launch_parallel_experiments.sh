#!/bin/bash

# Parallel Multi-Model Launcher
# Launches experiments on multiple GPUs simultaneously for multiple models
# Usage: ./launch_parallel_experiments.sh "Model1" "Model2" "Model3" ...

# Check if model names are provided
if [ $# -eq 0 ]; then
    echo "❌ Error: No model names provided!"
    echo "Usage: $0 \"Model1\" \"Model2\" \"Model3\" ..."
    echo "Example: $0 \"ResNet-18\" \"EfficientNet-B0\" \"DenseNet-121\""
    exit 1
fi

# Get model names from command line arguments
MODEL_NAMES=("$@")
MAX_PARALLEL_MODELS=6

echo "🚀 LAUNCHING PARALLEL MULTI-MODEL EXPERIMENTS"
echo "=============================================================="
echo "🎯 Models: ${MODEL_NAMES[*]}"
echo "🔧 Regularization: Dropout 0.3, Weight Decay 1e-04"
echo "💾 Memory requirement: 6-8GB per experiment"
echo "🖥️ Available GPUs: 6 (GPUs 1, 2, 3, 4, 5, 6, 7, 8, 9)"
echo "📊 Learning rates: [1e-5, 1e-4, 1e-3] = 3 experiments per model"
echo "🚀 Max parallel models: $MAX_PARALLEL_MODELS"
echo ""

# Configuration
PYTHON_PATH="/lfs/turing1/0/mahmedc/miniconda3/envs/mahmedc_env/bin/python"
AVAILABLE_GPUS=(1 2 3 4 5 6 7 8 9)
LEARNING_RATES=(1e-5 1e-4 1e-3 1e-2)

# Function to launch experiments for a single model
launch_model_experiments() {
    local model_name="$1"
    local start_gpu_idx="$2"
    
    echo "🏃 Starting experiments for model: $model_name"
    
    # Base command template
    BASE_CMD="$PYTHON_PATH run_experiments.py \
        --config_csv experimentation_plan_simplified.csv \
        --biomarker_config config/biomarker_config_mlhc_wo_gender.yaml \
        --data_dir /lfs/turing1/0/mahmedc/Comorbidities-Detection/datasets/full_data \
        --model_name \"$model_name\" \
        --output_base_dir /lfs/turing1/0/mahmedc/Comorbidities-Detection/models/mlhc_target_wo_gender/linear_probe/regularized \
        --resume \
        --no_confirm \
        --epochs 100"
    
    # Launch experiments for each learning rate on different GPUs
    for i in "${!LEARNING_RATES[@]}"; do
        LR="${LEARNING_RATES[$i]}"
        GPU_IDX=$((start_gpu_idx + i))
        GPU="${AVAILABLE_GPUS[$GPU_IDX]}"
        
        # Create safe session name (replace spaces and special chars with underscores)
        SAFE_MODEL_NAME=$(echo "$model_name" | sed 's/[^a-zA-Z0-9_-]/_/g')
        SESSION_NAME="${SAFE_MODEL_NAME}_mlhc_targets_linear_probe_regularized_lr_${LR}_gpu_${GPU}"
        
        echo "  📋 GPU $GPU: Model $model_name, Learning rate $LR (tmux: $SESSION_NAME)"
        
        # Create tmux session and run experiment
        tmux new-session -d -s "$SESSION_NAME" -c "/lfs/turing1/0/mahmedc/Comorbidities-Detection/CT-Disease-Detection"
        
        # Send commands to tmux session
        tmux send-keys -t "$SESSION_NAME" "export CUDA_VISIBLE_DEVICES=$GPU" Enter
        tmux send-keys -t "$SESSION_NAME" "export HF_HOME=/lfs/turing1/0/mahmedc/.cache/huggingface" Enter
        tmux send-keys -t "$SESSION_NAME" "export HF_HUB_CACHE=/lfs/turing1/0/mahmedc/.cache/huggingface/hub" Enter
        tmux send-keys -t "$SESSION_NAME" "echo \"🚀 Starting $model_name MLHC Targets without Gender Linear Probe Regularized - LR: $LR, GPU: $GPU\"" Enter
        tmux send-keys -t "$SESSION_NAME" "$BASE_CMD --learning_rate $LR" Enter
        
        sleep 2  # Brief delay between launches
    done
}

# Launch experiments for each model, up to MAX_PARALLEL_MODELS at a time
current_gpu_offset=0
for model_idx in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$model_idx]}"
    
    # Check if we can launch this model (need 4 GPUs per model for 4 learning rates)
    if [ $current_gpu_offset -ge ${#AVAILABLE_GPUS[@]} ]; then
        echo "⚠️  Warning: Not enough GPUs available for model $model_name. Skipping..."
        continue
    fi
    
    # Launch experiments for this model
    launch_model_experiments "$model_name" $current_gpu_offset
    
    # Update GPU offset for next model (4 GPUs per model)
    current_gpu_offset=$((current_gpu_offset + 4))
    
    # Check if we've reached the maximum parallel models limit
    if [ $((model_idx + 1)) -ge $MAX_PARALLEL_MODELS ]; then
        echo "🛑 Reached maximum parallel models limit ($MAX_PARALLEL_MODELS). Stopping."
        break
    fi
done

echo ""
echo "✅ All experiments launched!"
echo ""
echo "📋 Monitor experiments:"
echo "  tmux list-sessions | grep mlhc_targets_linear_probe_regularized"
echo ""
echo "🔍 Attach to specific experiment:"
# Show all launched sessions
for model_idx in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$model_idx]}"
    SAFE_MODEL_NAME=$(echo "$model_name" | sed 's/[^a-zA-Z0-9_-]/_/g')
    
    echo "  # Model: $model_name"
    for i in "${!LEARNING_RATES[@]}"; do
        LR="${LEARNING_RATES[$i]}"
        GPU_IDX=$((model_idx * 4 + i))
        if [ $GPU_IDX -lt ${#AVAILABLE_GPUS[@]} ]; then
            GPU="${AVAILABLE_GPUS[$GPU_IDX]}"
            SESSION_NAME="${SAFE_MODEL_NAME}_mlhc_targets_linear_probe_regularized_lr_${LR}_gpu_${GPU}"
            echo "    tmux attach -t $SESSION_NAME  # LR: $LR, GPU: $GPU"
        fi
    done
    echo ""
done
echo "📊 Check GPU usage:"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "📁 Results will be saved in: /lfs/turing1/0/mahmedc/Comorbidities-Detection/models/mlhc_targets/linear_probe/regularized"
echo ""
echo "🛑 To stop all experiments:"
echo "  tmux list-sessions | grep mlhc_targets_linear_probe_regularized | cut -d: -f1 | xargs -I {} tmux kill-session -t {}"

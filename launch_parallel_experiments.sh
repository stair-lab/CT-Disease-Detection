#!/bin/bash

# Parallel Multi-Model Launcher with GradNorm Support
# Launches experiments on multiple GPUs simultaneously for multiple models
# Usage: ./launch_parallel_experiments.sh [OPTIONS] "Model1" "Model2" "Model3" ...

# Default values
USE_GRADNORM=false
GRADNORM_ALPHA=0.16
GRADNORM_UPDATE_FREQ=10

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --use-gradnorm)
            USE_GRADNORM=true
            shift
            ;;
        --gradnorm-alpha)
            GRADNORM_ALPHA="$2"
            shift 2
            ;;
        --gradnorm-update-freq)
            GRADNORM_UPDATE_FREQ="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] \"Model1\" \"Model2\" \"Model3\" ..."
            echo ""
            echo "Options:"
            echo "  --use-gradnorm              Enable GradNorm for loss balancing"
            echo "  --gradnorm-alpha FLOAT      GradNorm restoring force strength (default: 0.16)"
            echo "  --gradnorm-update-freq INT  Update GradNorm weights every N iterations (default: 10)"
            echo "  --help, -h                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 \"ResNet-18\" \"EfficientNet-B0\""
            echo "  $0 --use-gradnorm \"ResNet-18\" \"EfficientNet-B0\""
            echo "  $0 --use-gradnorm --gradnorm-alpha 0.12 --gradnorm-update-freq 5 \"ResNet-18\""
            exit 0
            ;;
        -*)
            echo "❌ Error: Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            # This is a model name
            MODEL_NAMES+=("$1")
            shift
            ;;
    esac
done

# Check if model names are provided
if [ ${#MODEL_NAMES[@]} -eq 0 ]; then
    echo "❌ Error: No model names provided!"
    echo "Usage: $0 [OPTIONS] \"Model1\" \"Model2\" \"Model3\" ..."
    echo "Use --help for more information"
    exit 1
fi
MAX_PARALLEL_MODELS=6

echo "🚀 LAUNCHING PARALLEL MULTI-MODEL EXPERIMENTS"
echo "=============================================================="
echo "🎯 Models: ${MODEL_NAMES[*]}"
echo "🔧 Regularization: Dropout 0.3, Weight Decay 1e-04"
if [ "$USE_GRADNORM" = true ]; then
    echo "🔄 GradNorm: ENABLED (alpha=$GRADNORM_ALPHA, update_freq=$GRADNORM_UPDATE_FREQ)"
else
    echo "🔄 GradNorm: DISABLED"
fi
echo "💾 Memory requirement: 6-8GB per experiment"
echo "🖥️ Available GPUs: 10 (GPUs 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)"
echo "📊 Learning rates: [1e-5, 1e-4, 1e-3, 1e-2] = 4 experiments per model"
echo "🚀 Max parallel models: $MAX_PARALLEL_MODELS"
echo ""

# Configuration
PYTHON_PATH="/lfs/turing1/0/mahmedc/miniconda3/envs/mahmedc_env/bin/python"
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7 8 9)
LEARNING_RATES=(1e-5 1e-4 1e-3 1e-2)

# Function to launch experiments for a single model
launch_model_experiments() {
    local model_name="$1"
    local start_gpu_idx="$2"
    
    echo "🏃 Starting experiments for model: $model_name"
    
    # Base command template
    BASE_CMD="$PYTHON_PATH run_experiments.py \
        --config_csv experimentation_plan_simplified.csv \
        --biomarker_config config/biomarker_config_mlhc.yaml \
        --data_dir /lfs/turing1/0/mahmedc/Comorbidities-Detection/datasets/full_data \
        --model_name \"$model_name\" \
        --output_base_dir /lfs/turing1/0/mahmedc/Comorbidities-Detection/models/mlhc_targets/grad_norm/linear_probe/regularized \
        --resume \
        --no_confirm \
        --epochs 100"
    
    # Add GradNorm parameters if enabled
    if [ "$USE_GRADNORM" = true ]; then
        BASE_CMD="$BASE_CMD --use_gradnorm --gradnorm_alpha $GRADNORM_ALPHA --gradnorm_update_freq $GRADNORM_UPDATE_FREQ"
    fi
    
    # Launch experiments for each learning rate on different GPUs
    for i in "${!LEARNING_RATES[@]}"; do
        LR="${LEARNING_RATES[$i]}"
        GPU_IDX=$((start_gpu_idx + i))
        GPU="${AVAILABLE_GPUS[$GPU_IDX]}"
        
        # Create safe session name (replace spaces and special chars with underscores)
        SAFE_MODEL_NAME=$(echo "$model_name" | sed 's/[^a-zA-Z0-9_-]/_/g')
        if [ "$USE_GRADNORM" = true ]; then
            SESSION_NAME="${SAFE_MODEL_NAME}_mlhc_targets_linear_probe_regularized_gradnorm_lr_${LR}_gpu_${GPU}"
            echo "  📋 GPU $GPU: Model $model_name, Learning rate $LR, GradNorm ON (tmux: $SESSION_NAME)"
        else
            SESSION_NAME="${SAFE_MODEL_NAME}_mlhc_targets_linear_probe_regularized_lr_${LR}_gpu_${GPU}"
            echo "  📋 GPU $GPU: Model $model_name, Learning rate $LR (tmux: $SESSION_NAME)"
        fi
        
        # Create tmux session and run experiment
        tmux new-session -d -s "$SESSION_NAME" -c "/lfs/turing1/0/mahmedc/Comorbidities-Detection/CT-Disease-Detection"
        
        # Send commands to tmux session
        tmux send-keys -t "$SESSION_NAME" "export CUDA_VISIBLE_DEVICES=$GPU" Enter
        tmux send-keys -t "$SESSION_NAME" "export HF_HOME=/lfs/turing1/0/mahmedc/.cache/huggingface" Enter
        tmux send-keys -t "$SESSION_NAME" "export HF_HUB_CACHE=/lfs/turing1/0/mahmedc/.cache/huggingface/hub" Enter
        if [ "$USE_GRADNORM" = true ]; then
            tmux send-keys -t "$SESSION_NAME" "echo \"🚀 Starting $model_name MLHC Targets Linear Probe Regularized with GradNorm - LR: $LR, GPU: $GPU, Alpha: $GRADNORM_ALPHA, Update Freq: $GRADNORM_UPDATE_FREQ\"" Enter
        else
            tmux send-keys -t "$SESSION_NAME" "echo \"🚀 Starting $model_name MLHC Targets Linear Probe Regularized - LR: $LR, GPU: $GPU\"" Enter
        fi
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
if [ "$USE_GRADNORM" = true ]; then
    echo "  tmux list-sessions | grep mlhc_targets_linear_probe_regularized_gradnorm"
else
    echo "  tmux list-sessions | grep mlhc_targets_linear_probe_regularized"
fi
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
            if [ "$USE_GRADNORM" = true ]; then
                SESSION_NAME="${SAFE_MODEL_NAME}_mlhc_targets_linear_probe_regularized_gradnorm_lr_${LR}_gpu_${GPU}"
                echo "    tmux attach -t $SESSION_NAME  # LR: $LR, GPU: $GPU, GradNorm ON"
            else
                SESSION_NAME="${SAFE_MODEL_NAME}_mlhc_targets_linear_probe_regularized_lr_${LR}_gpu_${GPU}"
                echo "    tmux attach -t $SESSION_NAME  # LR: $LR, GPU: $GPU"
            fi
        fi
    done
    echo ""
done
echo "📊 Check GPU usage:"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "📁 Results will be saved in: /lfs/turing1/0/mahmedc/Comorbidities-Detection/models/mlhc_targets/grad_norm/linear_probe/regularized"
echo ""
echo "🛑 To stop all experiments:"
if [ "$USE_GRADNORM" = true ]; then
    echo "  tmux list-sessions | grep mlhc_targets_linear_probe_regularized_gradnorm | cut -d: -f1 | xargs -I {} tmux kill-session -t {}"   
else
    echo "  tmux list-sessions | grep mlhc_targets_linear_probe_regularized | cut -d: -f1 | xargs -I {} tmux kill-session -t {}"
fi

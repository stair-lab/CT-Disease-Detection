"""
Gradient Weighted Class Activation Mapping (Grad-CAM) for Medical Image Analysis
Specifically designed for single-task models in the comorbidity detection system
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional, Union
import argparse
from tqdm import tqdm

from test import load_checkpoint, create_model_from_checkpoint, create_test_transforms, CustomCSVDataset
from config.biomarker_config import FlexibleBiomarkerConfig


class GradCAM:
    """
    Gradient Weighted Class Activation Mapping for single-task models
    """
    
    def __init__(self, model: torch.nn.Module, target_layer_name: str, device: str = 'cuda'):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained model
            target_layer_name: Name of the target convolutional layer
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.target_layer_name = target_layer_name
        
        # Get the target layer
        self.target_layer = self._get_target_layer()
        
        # Register hooks
        self.gradients = None
        self.activations = None
        self.handlers = []
        
        self._register_hooks()
    
    def _get_target_layer(self):
        """Get the target layer by name"""
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                return module
        
        # If exact name not found, try to find the last convolutional layer
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append((name, module))
        
        if conv_layers:
            print(f"⚠️  Target layer '{self.target_layer_name}' not found. Using last conv layer: {conv_layers[-1][0]}")
            return conv_layers[-1][1]
        
        raise ValueError(f"Could not find target layer '{self.target_layer_name}' or any convolutional layer")
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Register hooks
        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self.handlers.append(self.target_layer.register_backward_hook(backward_hook))
    
    def _remove_hooks(self):
        """Remove registered hooks"""
        for handler in self.handlers:
            handler.remove()
        self.handlers = []
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            class_idx: Class index to generate CAM for (if None, uses highest prediction)
            
        Returns:
            Grad-CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_()
        
        # Get prediction
        output = self.model(input_tensor)
        
        if class_idx is None:
            # For binary classification, use the single output
            if output.shape[1] == 1:
                class_idx = 0
            else:
                # For multiclass, use the highest prediction
                class_idx = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        if output.shape[1] == 1:
            # Binary classification - use the single output
            score = output[0, 0]
        else:
            # Multiclass - use the specific class
            score = output[0, class_idx]
        
        score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def __del__(self):
        """Clean up hooks when object is destroyed"""
        self._remove_hooks()


class GradCAMVisualizer:
    """
    Visualization utilities for Grad-CAM
    """
    
    def __init__(self, colormap: str = 'jet'):
        """
        Initialize visualizer
        
        Args:
            colormap: Matplotlib colormap name
        """
        self.colormap = colormap
    
    def overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Overlay heatmap on original image - EXACT original working logic
        
        Args:
            image: Original image [H, W] or [H, W, C]
            heatmap: Grad-CAM heatmap [H, W]
            alpha: Transparency of heatmap
            
        Returns:
            Overlaid image
        """
        # Resize heatmap to match image using PIL
        if heatmap.shape != image.shape[:2]:
            # Convert heatmap to PIL Image, resize, then back to numpy
            heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
            heatmap_pil = heatmap_pil.resize((image.shape[1], image.shape[0]), Image.LANCZOS)
            heatmap = np.array(heatmap_pil) / 255.0
        
        # Apply colormap to heatmap
        cmap = cm.get_cmap(self.colormap)
        heatmap_colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
        
        # Ensure image is in [0, 1] range
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Overlay - EXACT original working approach
        overlaid = alpha * heatmap_colored + (1 - alpha) * image
        
        return np.clip(overlaid, 0, 1)
    
    def create_visualization(self, image: np.ndarray, heatmap: np.ndarray, 
                           prediction: float, target_name: str = "HCC18", 
                           ground_truth: float = None, gt_status: str = None, 
                           save_path: Optional[str] = None, alpha: float = 0.4) -> plt.Figure:
        """
        Create 2-panel visualization: original and overlay
        
        Args:
            image: Original image
            heatmap: Grad-CAM heatmap
            prediction: Model prediction probability
            target_name: Name of the target biomarker
            ground_truth: Ground truth value from CSV
            save_path: Path to save the figure
            alpha: Transparency of heatmap overlay
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Set font to Times New Roman with better fallback handling
        import matplotlib.font_manager as fm
        
        # Try to find Times New Roman font
        times_fonts = [f.name for f in fm.fontManager.ttflist if 'times' in f.name.lower() or 'new roman' in f.name.lower()]
        
        if times_fonts:
            plt.rcParams['font.family'] = times_fonts[0]
            print(f"Using font: {times_fonts[0]}")
        else:
            # Fallback to serif font family
            plt.rcParams['font.family'] = 'serif'
            print("Times New Roman not found, using serif font family")
        
        plt.rcParams['font.size'] = 12
        
        # Determine prediction status using 0.9 threshold
        pred_status = "Present" if prediction > 0.9 else "Absent"
        
        if gt_status is not None:
            gt_status_print = "Present" if gt_status.lower() == "present" else "Absent"
        # Create title with prediction, ground truth, and status
        title_info = f'Prediction: {pred_status}'
        if ground_truth is not None and gt_status is not None:
            title_info += f'\nGround Truth: {gt_status_print}'
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title(f'Input CT-derived Surface Mesh\n{title_info}')
        axes[0].axis('off')
        
        # Plot heatmap
        # axes[1].imshow(heatmap, cmap=self.colormap)
        # axes[1].set_title('Grad-CAM Heatmap')
        # axes[1].axis('off')
        
        # Overlay
        overlaid = self.overlay_heatmap(image, heatmap, alpha=alpha)
        axes[1].imshow(overlaid)
        axes[1].set_title(f'Grad-CAM Overlay \n{target_name} Attention')
        axes[1].axis('off')
        
        # Add colorbar for heatmap reference
        im = axes[1].imshow(overlaid)
        cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label('Attention Intensity', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        return fig


def analyze_hcc18_model(checkpoint_path: str, biomarker_config_path: str, 
                       csv_path: str, output_dir: str = "gradcam_results",
                       num_samples: int = 10, target_layer: str = "layer4.1.conv2"):
    """
    Analyze HCC18 predictions using Grad-CAM
    
    Args:
        checkpoint_path: Path to model checkpoint
        biomarker_config_path: Path to biomarker configuration
        csv_path: Path to CSV file (e.g., test.csv, val.csv)
        output_dir: Output directory for results
        num_samples: Number of samples to analyze
        target_layer: Target layer for Grad-CAM
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load biomarker configuration
    print("Loading biomarker configuration...")
    biomarker_config = FlexibleBiomarkerConfig(biomarker_config_path)
    
    # Check if HCC18 is configured
    hcc18_biomarkers = [b for b in biomarker_config.binary_biomarkers if b.name == "HCC18"]
    if not hcc18_biomarkers:
        print("❌ HCC18 biomarker not found in configuration!")
        return
    
    print(f"✅ Found HCC18 biomarker: {hcc18_biomarkers[0]}")
    
    # Load checkpoint and model
    print("Loading model...")
    checkpoint = load_checkpoint(checkpoint_path)
    model, config = create_model_from_checkpoint(checkpoint, biomarker_config)
    
    # Extract data directory from CSV path
    data_dir = os.path.dirname(csv_path)
    csv_filename = os.path.basename(csv_path)
    
    print(f"Data directory: {data_dir}")
    print(f"CSV file: {csv_filename}")
    
    # Create test dataset
    print("Creating test dataset...")
    transform = create_test_transforms(config)
    test_dataset = CustomCSVDataset(
        data_dir, 
        biomarker_config, 
        transforms=transform, 
        size=256, 
        csv_file=csv_filename
    )
    
    # Filter for samples with HCC18 data
    hcc18_samples = []
    for i in range(len(test_dataset)):
        target = test_dataset.targets[i]
        layout = biomarker_config.get_tensor_layout()["HCC18"]
        hcc18_value = target[layout.start_idx]
        if not np.isnan(hcc18_value):  # Has valid HCC18 label
            hcc18_samples.append(i)
    
    print(f"Found {len(hcc18_samples)} samples with HCC18 labels")
    
    if len(hcc18_samples) == 0:
        print("❌ No samples with HCC18 labels found!")
        return
    
    # Select samples to analyze
    if num_samples > len(hcc18_samples):
        num_samples = len(hcc18_samples)
    
    selected_indices = np.random.choice(hcc18_samples, num_samples, replace=False)
    
    # Initialize Grad-CAM
    print(f"Initializing Grad-CAM for layer: {target_layer}")
    gradcam = GradCAM(model, target_layer)
    visualizer = GradCAMVisualizer()
    
    # Analyze samples
    print(f"Analyzing {num_samples} samples...")
    
    results = []
    
    for i, idx in enumerate(tqdm(selected_indices)):
        # Get sample
        image, target = test_dataset[idx]
        image_tensor = image.unsqueeze(0)  # Add batch dimension
        
        # Get ground truth
        layout = biomarker_config.get_tensor_layout()["HCC18"]
        gt_hcc18 = target[layout.start_idx].item()
        
        # Get prediction
        with torch.no_grad():
            prediction = model(image_tensor.to('cuda'))
            pred_hcc18 = torch.sigmoid(prediction[0, layout.start_idx]).item()
        
        # Get original filename from dataset
        original_filename = test_dataset.at(idx)
        
        # Get ground truth from test.csv
        gt_hcc18_raw = test_dataset.df.iloc[idx]['HCC18']
        # Convert string values to numeric (PRESENT=1, ABSENT=0) for calculations
        if isinstance(gt_hcc18_raw, str):
            gt_hcc18 = 1.0 if gt_hcc18_raw == "PRESENT" else 0.0
            gt_status = gt_hcc18_raw  # Use the original string value
        else:
            gt_hcc18 = float(gt_hcc18_raw)
            gt_status = "PRESENT" if gt_hcc18 > 0.5 else "ABSENT"
        
        # Convert image to numpy for visualization
        # Handle both single-channel and 3-channel images
        if image.shape[0] == 3:  # 3-channel image (RGB)
            # Convert to grayscale by taking the first channel or averaging
            image_np = image[0].numpy()  # Take first channel (they should be identical after grayscale->RGB conversion)
        else:  # Single channel
            image_np = image.squeeze().numpy()
        
        # Generate Grad-CAM
        heatmap = gradcam.generate_cam(image_tensor, class_idx=0)
        
        # Debug: Print heatmap statistics and ground truth info
        print(f"Sample {i}: Heatmap stats - min: {heatmap.min():.4f}, max: {heatmap.max():.4f}, mean: {heatmap.mean():.4f}")
        print(f"  Ground truth: {gt_hcc18_raw} -> {gt_hcc18} (type: {type(gt_hcc18)})")
        print(f"  Image stats - min: {image_np.min():.4f}, max: {image_np.max():.4f}, mean: {image_np.mean():.4f}")
        
        # Create visualization with customizable transparency
        fig = visualizer.create_visualization(
            image_np, heatmap, pred_hcc18, "HCC18", gt_hcc18, gt_status,
            save_path=os.path.join(output_dir, f"{original_filename}_heatmap.png"),
            alpha=0.4  # Adjust transparency (0.0 = no overlay, 1.0 = full overlay)
        )
        plt.close(fig)
        
        # Store results with 0.9 threshold
        results.append({
            'sample_idx': idx,
            'original_filename': original_filename,
            'ground_truth': gt_hcc18,
            'prediction': pred_hcc18,
            'correct_0.5': (pred_hcc18 > 0.5) == (gt_hcc18 > 0.5),
            'correct_0.9': (pred_hcc18 > 0.9) == (gt_hcc18 > 0.5),
            'heatmap_max': heatmap.max(),
            'heatmap_mean': heatmap.mean()
        })
    
    # Save results summary
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, "hcc18_gradcam_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("HCC18 Grad-CAM Analysis Summary")
    print(f"{'='*60}")
    print(f"Analyzed samples: {len(results)}")
    print(f"Accuracy (0.5 threshold): {results_df['correct_0.5'].mean():.3f}")
    print(f"Accuracy (0.9 threshold): {results_df['correct_0.9'].mean():.3f}")
    print(f"Average prediction: {results_df['prediction'].mean():.3f}")
    print(f"Average ground truth: {results_df['ground_truth'].mean():.3f}")
    print(f"Average heatmap intensity: {results_df['heatmap_mean'].mean():.3f}")
    print(f"Results saved to: {results_path}")
    print(f"Visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM Analysis for HCC18')
    parser.add_argument('--checkpoint_path', required=True, help='Path to model checkpoint')
    parser.add_argument('--biomarker_config', required=True, help='Path to biomarker configuration')
    parser.add_argument('--csv_path', required=True, help='Path to CSV file (e.g., /path/to/test.csv)')
    parser.add_argument('--output_dir', default='gradcam_results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to analyze')
    parser.add_argument('--target_layer', default='layer4.1.conv2', help='Target layer for Grad-CAM')
    
    args = parser.parse_args()
    
    analyze_hcc18_model(
        checkpoint_path=args.checkpoint_path,
        biomarker_config_path=args.biomarker_config,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        target_layer=args.target_layer
    )


if __name__ == "__main__":
    main()

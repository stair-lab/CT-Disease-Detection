"""
Model Factory for Multi-Task Comorbidity Detection
Supports all architectures from the experimentation plan
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    DenseNet121_Weights, EfficientNet_B0_Weights, EfficientNet_B4_Weights,
    ConvNeXt_Base_Weights, ViT_B_16_Weights, ViT_L_16_Weights,
    Swin_B_Weights
)

# MaxVit_B_Weights might not be available in older torchvision versions
try:
    from torchvision.models import MaxVit_B_Weights
except ImportError:
    MaxVit_B_Weights = None
import timm
from transformers import (
    CLIPModel, CLIPProcessor, 
    AutoModel, AutoProcessor,
    BlipModel, BlipProcessor
)
from .resnet34 import ResNet34
from .cc_resnet import resnet34


# Legacy MultiTaskHead - kept for backward compatibility
# Use FlexibleMultiTaskHead for new experiments
class MultiTaskHead(nn.Module):
    """Legacy multi-task head for comorbidity detection - use FlexibleMultiTaskHead instead"""
    
    def __init__(self, input_dim, num_binary_tasks=7, num_calcium_classes=4, 
                 num_regression_tasks=2, dropout=0.1):
        super().__init__()
        
        self.num_binary_tasks = num_binary_tasks
        self.num_calcium_classes = num_calcium_classes
        self.num_regression_tasks = num_regression_tasks
        
        # Shared feature processing
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(512)
        )
        
        # Task-specific heads
        self.binary_head = nn.Linear(512, num_binary_tasks)  # Binary classification
        self.calcium_head = nn.Linear(512, num_calcium_classes)  # Multiclass classification
        self.regression_head = nn.Linear(512, num_regression_tasks)  # Regression (age, RAF)
        
    def forward(self, x):
        shared_features = self.shared_layers(x)
        
        binary_out = self.binary_head(shared_features)  # [B, 7]
        calcium_out = self.calcium_head(shared_features)  # [B, 4]
        regression_out = self.regression_head(shared_features)  # [B, 2]
        
        # Concatenate all outputs
        return torch.cat([binary_out, calcium_out, regression_out], dim=1)


class ModelFactory:
    """Factory class to create models based on architecture name"""
    
    @staticmethod
    def create_model(architecture, num_classes=13, pretrained_weights=None, 
                    fine_tuning_strategy="full", dropout=0.1, biomarker_config=None, **kwargs):
        """
        Create model based on architecture specification
        
        Args:
            architecture: Model architecture name
            num_classes: Total number of output classes (flexible based on biomarker_config)
            pretrained_weights: Pretrained weights source
            fine_tuning_strategy: 'full', 'linear_probe', or 'partial'
            dropout: Dropout rate
            biomarker_config: FlexibleBiomarkerConfig instance (if None, uses legacy MultiTaskHead)
        """
        
        if architecture == "ResNet-18":
            return ModelFactory._create_resnet18(num_classes, pretrained_weights, 
                                                fine_tuning_strategy, dropout, biomarker_config)
        
        elif architecture == "ResNet-34":
            return ModelFactory._create_resnet34(num_classes, pretrained_weights, 
                                                fine_tuning_strategy, dropout, biomarker_config)
        
        elif architecture == "DenseNet-121":
            return ModelFactory._create_densenet121(num_classes, pretrained_weights, 
                                                   fine_tuning_strategy, dropout)
        
        elif architecture == "EfficientNet-B0":
            return ModelFactory._create_efficientnet_b0(num_classes, pretrained_weights, 
                                                       fine_tuning_strategy, dropout)
        
        elif architecture == "EfficientNet-B4":
            return ModelFactory._create_efficientnet_b4(num_classes, pretrained_weights, 
                                                       fine_tuning_strategy, dropout)
        
        elif architecture == "ConvNeXt-Base":
            return ModelFactory._create_convnext_base(num_classes, pretrained_weights, 
                                                     fine_tuning_strategy, dropout)
        
        elif architecture in ["ViT-Small (DINOv2)", "ViT-Base (DINOv2)", "ViT-Large (DINOv2)"]:
            return ModelFactory._create_dinov2_vit(architecture, num_classes, 
                                                  fine_tuning_strategy, dropout)
        
        elif architecture == "Swin Transformer-Base":
            return ModelFactory._create_swin_base(num_classes, pretrained_weights, 
                                                 fine_tuning_strategy, dropout)
        
        elif architecture == "MaxViT-Base":
            return ModelFactory._create_maxvit_base(num_classes, pretrained_weights, 
                                                   fine_tuning_strategy, dropout)
        
        elif architecture in ["CLIP-ViT-B/16 (full fine-tuning)", "CLIP-ViT-B/16 (frozen linear probe)"]:
            return ModelFactory._create_clip_vit_b16(architecture, num_classes, 
                                                    fine_tuning_strategy, dropout)
        
        elif architecture == "CLIP-ViT-L/14 (full fine-tuning)":
            return ModelFactory._create_clip_vit_l14(num_classes, fine_tuning_strategy, dropout)
        
        elif architecture == "BLIP-2 ViT-Base":
            return ModelFactory._create_blip2_vit_base(num_classes, fine_tuning_strategy, dropout)
        
        elif architecture in ["MedGemma", "MedCLIP", "BiomedCLIP"]:
            return ModelFactory._create_medical_vlm(architecture, num_classes, 
                                                   fine_tuning_strategy, dropout)
        
        elif architecture in ["Stable Diffusion v1.5 VAE Encoder", 
                             "Stable Diffusion v1.5 VAE Encoder (frozen)",
                             "Stable Diffusion XL VAE Encoder"]:
            return ModelFactory._create_diffusion_encoder(architecture, num_classes, 
                                                         fine_tuning_strategy, dropout)
        
        elif architecture == "DiT-Base (Diffusion Transformer)":
            return ModelFactory._create_dit_base(num_classes, fine_tuning_strategy, dropout)
        
        elif architecture == "MAE ViT-Base (self-supervised)":
            return ModelFactory._create_mae_vit_base(num_classes, fine_tuning_strategy, dropout)
        
        elif architecture == "ResNet-50 (RadImageNet)":
            return ModelFactory._create_resnet50_radimgnet(num_classes, fine_tuning_strategy, dropout)
        
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    @staticmethod
    def _create_multitask_head(feature_dim, dropout, biomarker_config):
        """Create appropriate multi-task head based on configuration"""
        if biomarker_config is not None:
            # Use flexible multi-task head
            from .flexible_multitask_head import FlexibleMultiTaskHead
            return FlexibleMultiTaskHead(feature_dim, biomarker_config, dropout=dropout)
        else:
            # Use legacy multi-task head for backward compatibility
            return MultiTaskHead(feature_dim, dropout=dropout)
    
    @staticmethod
    def _create_resnet18(num_classes, pretrained_weights, fine_tuning_strategy, dropout, biomarker_config):
        if pretrained_weights == "ImageNet":
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            # Keep 3-channel input for pretrained weights, we'll convert images to 3-channel
        else:
            model = models.resnet18(weights=None)
            # For non-pretrained, we can use single channel
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace classifier with flexible multi-task head
        feature_dim = model.fc.in_features
        model.fc = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config)
        
        if fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_resnet34(num_classes, pretrained_weights, fine_tuning_strategy, dropout, biomarker_config):
        if pretrained_weights == "ImageNet":
            model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            # Keep 3-channel input for pretrained weights
            feature_dim = model.fc.in_features
            model.fc = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config)
        else:
            # Use existing custom ResNet34 with flexible head
            if biomarker_config is not None:
                # Create model without final classifier, then add flexible head
                model = ResNet34(num_classes=1)  # Temporary
                # Replace with flexible head
                feature_dim = model.fc.in_features if hasattr(model, 'fc') else 512
                model.fc = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config)
            else:
                # Legacy behavior
                model = ResNet34(num_classes=num_classes)
        
        if fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_densenet121(num_classes, pretrained_weights, fine_tuning_strategy, dropout):
        if pretrained_weights == "ImageNet":
            model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            model = models.densenet121(weights=None)
        
        # Keep 3-channel input for pretrained weights
        # model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace classifier
        feature_dim = model.classifier.in_features
        model.classifier = MultiTaskHead(feature_dim, dropout=dropout)
        
        if fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_efficientnet_b0(num_classes, pretrained_weights, fine_tuning_strategy, dropout):
        if pretrained_weights == "ImageNet":
            model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b0(weights=None)
        
        # Keep 3-channel input for pretrained weights
        # model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Replace classifier
        feature_dim = model.classifier[1].in_features
        model.classifier = MultiTaskHead(feature_dim, dropout=dropout)
        
        if fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_efficientnet_b4(num_classes, pretrained_weights, fine_tuning_strategy, dropout):
        if pretrained_weights == "ImageNet":
            model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b4(weights=None)
        
        # Modify for single channel input
        model.features[0][0] = nn.Conv2d(1, 48, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Replace classifier
        feature_dim = model.classifier[1].in_features
        model.classifier = MultiTaskHead(feature_dim, dropout=dropout)
        
        if fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_convnext_base(num_classes, pretrained_weights, fine_tuning_strategy, dropout):
        if pretrained_weights == "ImageNet-22K":
            # Use IMAGENET1K_V1 as IMAGENET22K_V1 is not available
            model = models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
            # Keep 3-channel input for pretrained weights, we'll convert images to 3-channel
        else:
            model = models.convnext_base(weights=None)
            # For non-pretrained, modify for single channel input
            model.features[0][0] = nn.Conv2d(1, 128, kernel_size=4, stride=4)
        
        # Replace classifier
        feature_dim = model.classifier[2].in_features
        model.classifier = nn.Sequential(
            model.classifier[0],  # LayerNorm
            model.classifier[1],  # Flatten
            MultiTaskHead(feature_dim, dropout=dropout)
        )
        
        if fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier[2].parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_dinov2_vit(architecture, num_classes, fine_tuning_strategy, dropout):
        # Use timm for DINOv2 models
        if "Small" in architecture:
            model_name = "vit_small_patch14_dinov2.lvd142m"
        elif "Base" in architecture:
            model_name = "vit_base_patch14_dinov2.lvd142m"
        else:  # Large
            model_name = "vit_large_patch14_dinov2.lvd142m"
        
        model = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=256)  # Remove head, set input size
        
        # Add custom multi-task head
        feature_dim = model.num_features
        model.head = MultiTaskHead(feature_dim, dropout=dropout)
        
        # Keep 3-channel input since training script converts images to 3-channel
        # No need to modify patch_embed for single channel
        
        if fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_swin_base(num_classes, pretrained_weights, fine_tuning_strategy, dropout):
        if pretrained_weights == "ImageNet-22K":
            model = models.swin_b(weights=Swin_B_Weights.IMAGENET22K_V1)
        else:
            model = models.swin_b(weights=None)
        
        # Modify for single channel input
        model.features[0][0] = nn.Conv2d(1, 128, kernel_size=4, stride=4)
        
        # Replace head
        feature_dim = model.head.in_features
        model.head = MultiTaskHead(feature_dim, dropout=dropout)
        
        if fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_maxvit_base(num_classes, pretrained_weights, fine_tuning_strategy, dropout):
        # Use timm for MaxViT
        model = timm.create_model('maxvit_base_tf_224.in1k', pretrained=True, num_classes=0)
        
        # Add custom multi-task head
        feature_dim = model.num_features
        model.head = MultiTaskHead(feature_dim, dropout=dropout)
        
        # Modify for single channel input - MaxViT uses stem
        if hasattr(model, 'stem') and hasattr(model.stem, 'conv1'):
            old_conv = model.stem.conv1
            model.stem.conv1 = nn.Conv2d(1, old_conv.out_channels,
                                        kernel_size=old_conv.kernel_size,
                                        stride=old_conv.stride,
                                        padding=old_conv.padding)
        
        if fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_clip_vit_b16(architecture, num_classes, fine_tuning_strategy, dropout):
        # Placeholder for CLIP implementation
        # This would require proper CLIP model loading and modification
        print(f"Warning: {architecture} not fully implemented yet. Using ViT-Base as placeholder.")
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        # Modify for single channel input
        old_conv = model.patch_embed.proj
        model.patch_embed.proj = nn.Conv2d(1, old_conv.out_channels,
                                          kernel_size=old_conv.kernel_size,
                                          stride=old_conv.stride,
                                          padding=old_conv.padding)
        
        feature_dim = model.num_features
        model.head = MultiTaskHead(feature_dim, dropout=dropout)
        
        if "frozen" in architecture or fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_clip_vit_l14(num_classes, fine_tuning_strategy, dropout):
        # Placeholder implementation
        print("Warning: CLIP-ViT-L/14 not fully implemented yet. Using ViT-Large as placeholder.")
        model = timm.create_model('vit_large_patch14_224', pretrained=True, num_classes=0)
        
        # Modify for single channel input
        old_conv = model.patch_embed.proj
        model.patch_embed.proj = nn.Conv2d(1, old_conv.out_channels,
                                          kernel_size=old_conv.kernel_size,
                                          stride=old_conv.stride,
                                          padding=old_conv.padding)
        
        feature_dim = model.num_features
        model.head = MultiTaskHead(feature_dim, dropout=dropout)
        
        return model
    
    @staticmethod
    def _create_blip2_vit_base(num_classes, fine_tuning_strategy, dropout):
        # Placeholder implementation
        print("Warning: BLIP-2 not fully implemented yet. Using ViT-Base as placeholder.")
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        # Modify for single channel input
        old_conv = model.patch_embed.proj
        model.patch_embed.proj = nn.Conv2d(1, old_conv.out_channels,
                                          kernel_size=old_conv.kernel_size,
                                          stride=old_conv.stride,
                                          padding=old_conv.padding)
        
        feature_dim = model.num_features
        model.head = MultiTaskHead(feature_dim, dropout=dropout)
        
        return model
    
    @staticmethod
    def _create_medical_vlm(architecture, num_classes, fine_tuning_strategy, dropout):
        # Placeholder for medical VLMs
        print(f"Warning: {architecture} not fully implemented yet. Using ViT-Base as placeholder.")
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        # Modify for single channel input
        old_conv = model.patch_embed.proj
        model.patch_embed.proj = nn.Conv2d(1, old_conv.out_channels,
                                          kernel_size=old_conv.kernel_size,
                                          stride=old_conv.stride,
                                          padding=old_conv.padding)
        
        feature_dim = model.num_features
        model.head = MultiTaskHead(feature_dim, dropout=dropout)
        
        return model
    
    @staticmethod
    def _create_diffusion_encoder(architecture, num_classes, fine_tuning_strategy, dropout):
        # Placeholder for diffusion model encoders
        print(f"Warning: {architecture} not fully implemented yet. Using ResNet-50 as placeholder.")
        model = models.resnet50(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        feature_dim = model.fc.in_features
        model.fc = MultiTaskHead(feature_dim, dropout=dropout)
        
        if "frozen" in architecture:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_dit_base(num_classes, fine_tuning_strategy, dropout):
        # Placeholder for DiT
        print("Warning: DiT-Base not fully implemented yet. Using ViT-Base as placeholder.")
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        # Modify for single channel input
        old_conv = model.patch_embed.proj
        model.patch_embed.proj = nn.Conv2d(1, old_conv.out_channels,
                                          kernel_size=old_conv.kernel_size,
                                          stride=old_conv.stride,
                                          padding=old_conv.padding)
        
        feature_dim = model.num_features
        model.head = MultiTaskHead(feature_dim, dropout=dropout)
        
        return model
    
    @staticmethod
    def _create_mae_vit_base(num_classes, fine_tuning_strategy, dropout):
        # Use timm for MAE
        model = timm.create_model('vit_base_patch16_224.mae', pretrained=True, num_classes=0)
        
        # Modify for single channel input
        old_conv = model.patch_embed.proj
        model.patch_embed.proj = nn.Conv2d(1, old_conv.out_channels,
                                          kernel_size=old_conv.kernel_size,
                                          stride=old_conv.stride,
                                          padding=old_conv.padding)
        
        feature_dim = model.num_features
        model.head = MultiTaskHead(feature_dim, dropout=dropout)
        
        return model
    
    @staticmethod
    def _create_resnet50_radimgnet(num_classes, fine_tuning_strategy, dropout):
        # Placeholder for RadImageNet weights
        print("Warning: RadImageNet weights not available. Using ImageNet ResNet-50.")
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify for single channel input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace classifier
        feature_dim = model.fc.in_features
        model.fc = MultiTaskHead(feature_dim, dropout=dropout)
        
        return model


def get_model_memory_requirement(architecture):
    """Get expected GPU memory requirement for architecture"""
    memory_map = {
        "ResNet-18": "4-6GB",
        "ResNet-34": "6-8GB", 
        "DenseNet-121": "8-10GB",
        "EfficientNet-B0": "6-8GB",
        "EfficientNet-B4": "12-16GB",
        "ConvNeXt-Base": "10-12GB",
        "ViT-Small (DINOv2)": "8-10GB",
        "ViT-Base (DINOv2)": "12-16GB",
        "ViT-Large (DINOv2)": "24-32GB",
        "Swin Transformer-Base": "12-16GB",
        "MaxViT-Base": "14-18GB",
        "CLIP-ViT-B/16 (full fine-tuning)": "12-16GB",
        "CLIP-ViT-B/16 (frozen linear probe)": "8-10GB",
        "CLIP-ViT-L/14 (full fine-tuning)": "24-32GB",
        "BLIP-2 ViT-Base": "16-20GB",
        "Stable Diffusion v1.5 VAE Encoder": "12-16GB",
        "Stable Diffusion v1.5 VAE Encoder (frozen)": "8-10GB",
        "Stable Diffusion XL VAE Encoder": "20-24GB",
        "DiT-Base (Diffusion Transformer)": "16-20GB",
        "MAE ViT-Base (self-supervised)": "12-16GB",
        "MedGemma": "16-24GB",
        "MedCLIP": "12-16GB",
        "BiomedCLIP": "12-16GB",
        "ResNet-50 (RadImageNet)": "6-8GB"
    }
    return memory_map.get(architecture, "Unknown")


if __name__ == "__main__":
    # Test model creation
    factory = ModelFactory()
    model = factory.create_model("ResNet-18", num_classes=13)
    print(f"Created ResNet-18 model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test input
    x = torch.randn(2, 1, 256, 256)  # Batch of 2 CT scans
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be [2, 13]

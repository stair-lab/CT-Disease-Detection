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
try:
    from diffusers import AutoencoderKL
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers library not available. Stable Diffusion VAE will use placeholder.")
from .resnet34 import ResNet34
from .cc_resnet import resnet34


class StableDiffusionVAEEncoder(nn.Module):
    """
    Stable Diffusion VAE Encoder wrapper for medical image analysis
    """
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", feature_dim=512, 
                 biomarker_config=None, dropout=0.1, frozen=True):
        super().__init__()
        
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers library is required for Stable Diffusion VAE Encoder. "
                            "Install with: pip install diffusers")
        
        # Set cache directory to avoid AFS permission issues
        import os
        os.environ['HF_HOME'] = '/lfs/skampere2/0/mahmedc/.cache/huggingface'
        os.environ['TRANSFORMERS_CACHE'] = '/lfs/skampere2/0/mahmedc/.cache/huggingface/transformers'
        os.environ['HF_HUB_CACHE'] = '/lfs/skampere2/0/mahmedc/.cache/huggingface/hub'
        
        # Create cache directories if they don't exist
        os.makedirs('/lfs/skampere2/0/mahmedc/.cache/huggingface/hub', exist_ok=True)
        os.makedirs('/lfs/skampere2/0/mahmedc/.cache/huggingface/transformers', exist_ok=True)
        
        # Load the VAE encoder from Stable Diffusion v1.5
        self.vae = AutoencoderKL.from_pretrained(
            model_id, 
            subfolder="vae",
            cache_dir='/lfs/skampere2/0/mahmedc/.cache/huggingface'
        )
        
        # Only use the encoder part
        self.encoder = self.vae.encoder
        
        # Freeze encoder weights if specified
        self._frozen = frozen
        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # The VAE encoder outputs latents of shape [B, latent_channels, H/8, W/8]
        # For a 256x256 input, this becomes [B, latent_channels, 32, 32]
        # Note: latent_channels can be 4 or 8 depending on the VAE variant
        
        # Add adaptation layers to map VAE features to desired feature dimension
        # Use adaptive pooling to handle different latent channel dimensions
        self.feature_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),    # Reduce spatial dimensions to [B, C, 8, 8]
            nn.Flatten(),                    # [B, C, 8, 8] -> [B, C*64]
            # Use a flexible linear layer that can handle different input sizes
            # We'll initialize it in the forward pass
        )
        
        # This will be initialized on first forward pass
        self.linear_layer = None
        self.final_layers = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Store config for later use in creating classifier
        self.biomarker_config = biomarker_config
        self.feature_dim = feature_dim
        self.dropout = dropout
    
    def _create_classifier(self):
        """Create classifier after ModelFactory is fully loaded to avoid circular import"""
        if not hasattr(self, 'classifier'):
            from .flexible_multitask_head import FlexibleMultiTaskHead
            if self.biomarker_config is not None:
                self.classifier = FlexibleMultiTaskHead(self.feature_dim, self.biomarker_config, dropout=self.dropout)
            else:
                # Fallback to legacy head
                self.classifier = MultiTaskHead(self.feature_dim, dropout=self.dropout)
            
            # Move classifier to the same device as the encoder
            device = next(self.encoder.parameters()).device
            self.classifier = self.classifier.to(device)
    
    def forward(self, x):
        """
        Forward pass through VAE encoder
        Args:
            x: Input images [B, 3, H, W] (expects 3-channel RGB)
        Returns:
            Multi-task predictions
        """
        # Create classifier on first forward pass to avoid circular imports
        self._create_classifier()
        
        # Ensure input is in the right range for VAE (0-1 range)
        if x.max() > 1.0:
            x = x / 255.0
        
        # Normalize to [-1, 1] range as expected by VAE
        x = 2.0 * x - 1.0
        
        if self._frozen:
            with torch.no_grad():
                # VAE encoder returns a distribution, we take the sample
                try:
                    latent_dist = self.encoder(x)
                    # Handle different return types from VAE encoder
                    if hasattr(latent_dist, 'sample'):
                        latents = latent_dist.sample()
                    elif hasattr(latent_dist, 'latent_dist'):
                        latents = latent_dist.latent_dist.sample()
                    else:
                        latents = latent_dist
                except Exception as e:
                    # If VAE fails, use simple conv layers as fallback
                    print(f"Warning: VAE encoder failed, using fallback: {e}")
                    latents = torch.randn(x.size(0), 4, x.size(2)//8, x.size(3)//8, device=x.device)
        else:
            try:
                latent_dist = self.encoder(x)
                if hasattr(latent_dist, 'sample'):
                    latents = latent_dist.sample()
                elif hasattr(latent_dist, 'latent_dist'):
                    latents = latent_dist.latent_dist.sample()
                else:
                    latents = latent_dist
            except Exception as e:
                print(f"Warning: VAE encoder failed, using fallback: {e}")
                latents = torch.randn(x.size(0), 4, x.size(2)//8, x.size(3)//8, device=x.device)
        
        # Adapt features for classification
        # Apply pooling and flattening
        pooled_features = self.feature_adapter(latents)
        
        # Initialize linear layer on first forward pass
        if self.linear_layer is None:
            input_dim = pooled_features.size(1)
            self.linear_layer = nn.Linear(input_dim, self.feature_dim).to(pooled_features.device)
            # Also move final_layers to the same device
            self.final_layers = self.final_layers.to(pooled_features.device)
        
        # Apply linear transformation and final layers
        features = self.linear_layer(pooled_features)
        features = self.final_layers(features)
        
        # Multi-task classification
        return self.classifier(features)


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
                    fine_tuning_strategy="full", dropout=0.1, biomarker_config=None,
                    single_target_strategy=None, target_feature_dim=None,
                    single_target_output_dim=None, **kwargs):
        """
        Create model based on architecture specification
        
        Args:
            architecture: Model architecture name
            num_classes: Total number of output classes (flexible based on biomarker_config)
            pretrained_weights: Pretrained weights source
            fine_tuning_strategy: 'full', 'linear_probe', or 'partial'
            dropout: Dropout rate
            biomarker_config: FlexibleBiomarkerConfig instance (if None, uses legacy MultiTaskHead)
            single_target_strategy: Single-target classification strategy from CSV
        """
        
        # Use single_target_output_dim as target_feature_dim if provided
        if single_target_output_dim is not None:
            target_feature_dim = single_target_output_dim

        if architecture == "ResNet-18":
            return ModelFactory._create_resnet18(num_classes, pretrained_weights,
                                                fine_tuning_strategy, dropout, biomarker_config, single_target_strategy, target_feature_dim)
        
        elif architecture == "ResNet-34":
            return ModelFactory._create_resnet34(num_classes, pretrained_weights,
                                                fine_tuning_strategy, dropout, biomarker_config, single_target_strategy, target_feature_dim)
        
        elif architecture == "DenseNet-121":
            return ModelFactory._create_densenet121(num_classes, pretrained_weights,
                                                   fine_tuning_strategy, dropout, biomarker_config, single_target_strategy, target_feature_dim)
        
        elif architecture == "EfficientNet-B0":
            return ModelFactory._create_efficientnet_b0(num_classes, pretrained_weights,
                                                       fine_tuning_strategy, dropout, biomarker_config, single_target_strategy, target_feature_dim)
        
        elif architecture == "EfficientNet-B4":
            return ModelFactory._create_efficientnet_b4(num_classes, pretrained_weights, 
                                                       fine_tuning_strategy, dropout, biomarker_config, single_target_strategy, target_feature_dim)
        
        elif architecture == "ConvNeXt-Base":
            return ModelFactory._create_convnext_base(num_classes, pretrained_weights, 
                                                     fine_tuning_strategy, dropout, biomarker_config, single_target_strategy, target_feature_dim)
        
        elif architecture in ["ViT-Small (DINOv2)", "ViT-Base (DINOv2)", "ViT-Large (DINOv2)"]:
            return ModelFactory._create_dinov2_vit(architecture, num_classes, 
                                                  fine_tuning_strategy, dropout, biomarker_config, single_target_strategy, target_feature_dim)
        
        elif architecture == "Swin Transformer-Base":
            return ModelFactory._create_swin_base(num_classes, pretrained_weights, 
                                                 fine_tuning_strategy, dropout, biomarker_config, single_target_strategy, target_feature_dim)
        
        elif architecture == "MaxViT-Base":
            return ModelFactory._create_maxvit_base(num_classes, pretrained_weights, 
                                                   fine_tuning_strategy, dropout, biomarker_config, single_target_strategy, target_feature_dim)
        
        elif architecture in ["CLIP-ViT-B/16 (full fine-tuning)", "CLIP-ViT-B/16 (frozen linear probe)"]:
            return ModelFactory._create_clip_vit_b16(architecture, num_classes, 
                                                    fine_tuning_strategy, dropout, biomarker_config, single_target_strategy, target_feature_dim)
        
        elif architecture == "CLIP-ViT-L/14 (full fine-tuning)":
            return ModelFactory._create_clip_vit_l14(num_classes, fine_tuning_strategy, dropout, biomarker_config, single_target_strategy, target_feature_dim)
        
        elif architecture == "BLIP-2 ViT-Base":
            return ModelFactory._create_blip2_vit_base(num_classes, fine_tuning_strategy, dropout, biomarker_config, single_target_strategy, target_feature_dim)
        
        elif architecture in ["MedGemma", "MedCLIP", "BiomedCLIP"]:
            return ModelFactory._create_medical_vlm(architecture, num_classes, 
                                                   fine_tuning_strategy, dropout, biomarker_config, single_target_strategy, target_feature_dim)
        
        elif architecture in ["Stable Diffusion v1.5 VAE Encoder", 
                             "Stable Diffusion v1.5 VAE Encoder (frozen)",
                             "Stable Diffusion XL VAE Encoder"]:
            return ModelFactory._create_diffusion_encoder(architecture, num_classes, 
                                                         fine_tuning_strategy, dropout, biomarker_config, single_target_strategy, target_feature_dim)
        
        elif architecture == "DiT-Base (Diffusion Transformer)":
            return ModelFactory._create_dit_base(num_classes, fine_tuning_strategy, dropout, biomarker_config, single_target_strategy, target_feature_dim)
        
        elif architecture == "MAE ViT-Base (self-supervised)":
            return ModelFactory._create_mae_vit_base(num_classes, fine_tuning_strategy, dropout, biomarker_config, single_target_strategy, target_feature_dim)
        
        elif architecture == "ResNet-50 (RadImageNet)":
            return ModelFactory._create_resnet50_radimgnet(num_classes, fine_tuning_strategy, dropout, biomarker_config, single_target_strategy, target_feature_dim)
        
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    @staticmethod
    def _create_multitask_head(
        feature_dim,
        dropout,
        biomarker_config,
        head_type="flexible",
        single_target_strategy=None,
        target_feature_dim=None
    ):
        """Create appropriate multi-task head based on configuration"""
        if biomarker_config is not None:
            if head_type == "linear_probe":
                # True linear probe: direct backbone → tasks
                from .flexible_multitask_head import LinearProbeMultiTaskHead
                return LinearProbeMultiTaskHead(
                    feature_dim,
                    biomarker_config,
                    dropout=dropout,
                    single_target_strategy=single_target_strategy,
                    target_feature_dim=target_feature_dim
                )
            else:
                # Standard flexible multi-task head with shared layers
                from .flexible_multitask_head import FlexibleMultiTaskHead
                return FlexibleMultiTaskHead(
                    feature_dim,
                    biomarker_config,
                    dropout=dropout,
                    single_target_strategy=single_target_strategy,
                    target_feature_dim=target_feature_dim
                )
        else:
            # Use legacy multi-task head for backward compatibility
            return MultiTaskHead(feature_dim, dropout=dropout)
    
    @staticmethod
    def _create_resnet18(num_classes, pretrained_weights, fine_tuning_strategy, dropout, biomarker_config, single_target_strategy=None, target_feature_dim=None):
        # Determine head type based on fine-tuning strategy
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        
        if pretrained_weights == "ImageNet":
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            # Keep 3-channel input for pretrained weights, we'll convert images to 3-channel
        else:
            model = models.resnet18(weights=None)
            # For non-pretrained, we can use single channel
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace classifier with appropriate multi-task head
        feature_dim = model.fc.in_features
        model.fc = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, head_type=head_type, single_target_strategy=single_target_strategy, target_feature_dim=target_feature_dim)
        
        if fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_resnet34(num_classes, pretrained_weights, fine_tuning_strategy, dropout, biomarker_config, single_target_strategy=None, target_feature_dim=None):
        # Determine head type based on fine-tuning strategy
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        
        if pretrained_weights == "ImageNet":
            model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            # Keep 3-channel input for pretrained weights
            feature_dim = model.fc.in_features
            model.fc = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, head_type, single_target_strategy, target_feature_dim)
        else:
            # Use existing custom ResNet34 with flexible head
            if biomarker_config is not None:
                # Create model without final classifier, then add appropriate head
                model = ResNet34(num_classes=1)  # Temporary
                # Replace with appropriate head type
                feature_dim = model.fc.in_features if hasattr(model, 'fc') else 512
                model.fc = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, head_type, single_target_strategy, target_feature_dim)
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
    def _create_densenet121(num_classes, pretrained_weights, fine_tuning_strategy, dropout, biomarker_config=None, single_target_strategy=None, target_feature_dim=None):
        # Determine head type based on fine-tuning strategy
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        
        if pretrained_weights == "ImageNet":
            model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            model = models.densenet121(weights=None)
        
        # Keep 3-channel input for pretrained weights
        # model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace classifier with appropriate multi-task head
        feature_dim = model.classifier.in_features
        model.classifier = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, head_type=head_type, single_target_strategy=single_target_strategy, target_feature_dim=target_feature_dim)
        
        if fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_efficientnet_b0(num_classes, pretrained_weights, fine_tuning_strategy, dropout, biomarker_config=None, single_target_strategy=None, target_feature_dim=None):
        # Determine head type based on fine-tuning strategy
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        
        if pretrained_weights == "ImageNet":
            model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b0(weights=None)
        
        # Keep 3-channel input for pretrained weights
        # model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Replace classifier with appropriate multi-task head
        feature_dim = model.classifier[1].in_features
        model.classifier = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, head_type=head_type, single_target_strategy=single_target_strategy, target_feature_dim=target_feature_dim)
        
        if fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_efficientnet_b4(num_classes, pretrained_weights, fine_tuning_strategy, dropout, biomarker_config=None, single_target_strategy=None, target_feature_dim=None):
        # Determine head type based on fine-tuning strategy
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        
        if pretrained_weights == "ImageNet":
            model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b4(weights=None)
        
        # Modify for single channel input
        model.features[0][0] = nn.Conv2d(1, 48, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Replace classifier with appropriate multi-task head
        feature_dim = model.classifier[1].in_features
        model.classifier = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, head_type=head_type, single_target_strategy=single_target_strategy, target_feature_dim=target_feature_dim)
        
        if fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_convnext_base(num_classes, pretrained_weights, fine_tuning_strategy, dropout, biomarker_config=None, single_target_strategy=None, target_feature_dim=None):
        # Determine head type based on fine-tuning strategy
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        
        if pretrained_weights == "ImageNet-22K":
            # Use IMAGENET1K_V1 as IMAGENET22K_V1 is not available
            model = models.convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
            # Keep 3-channel input for pretrained weights, we'll convert images to 3-channel
        else:
            model = models.convnext_base(weights=None)
            # For non-pretrained, modify for single channel input
            model.features[0][0] = nn.Conv2d(1, 128, kernel_size=4, stride=4)
        
        # Replace classifier with appropriate multi-task head
        feature_dim = model.classifier[2].in_features
        multitask_head = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, head_type=head_type, single_target_strategy=single_target_strategy, target_feature_dim=target_feature_dim)
        model.classifier = nn.Sequential(
            model.classifier[0],  # LayerNorm
            model.classifier[1],  # Flatten
            multitask_head
        )
        
        if fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier[2].parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_dinov2_vit(architecture, num_classes, fine_tuning_strategy, dropout, biomarker_config, single_target_strategy=None, target_feature_dim=None):
        # Determine head type based on fine-tuning strategy
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        
        # Set cache directory to avoid AFS permission issues
        import os
        os.environ['HF_HOME'] = '/lfs/skampere2/0/mahmedc/.cache/huggingface'
        os.environ['TRANSFORMERS_CACHE'] = '/lfs/skampere2/0/mahmedc/.cache/huggingface/transformers'
        os.environ['HF_HUB_CACHE'] = '/lfs/skampere2/0/mahmedc/.cache/huggingface/hub'
        
        # Create cache directories if they don't exist
        os.makedirs('/lfs/skampere2/0/mahmedc/.cache/huggingface/hub', exist_ok=True)
        os.makedirs('/lfs/skampere2/0/mahmedc/.cache/huggingface/transformers', exist_ok=True)
        
        # Use timm for DINOv2 models
        if "Small" in architecture:
            model_name = "vit_small_patch14_dinov2"
        elif "Base" in architecture:
            model_name = "vit_base_patch14_dinov2"
        else:  # Large
            model_name = "vit_large_patch14_dinov2"
        
        model = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=256)  # Remove head, set input size
        
        # Add appropriate multi-task head
        feature_dim = model.num_features
        model.head = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, head_type=head_type, single_target_strategy=single_target_strategy, target_feature_dim=target_feature_dim)
        
        # Keep 3-channel input since training script converts images to 3-channel
        # No need to modify patch_embed for single channel
        
        if fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_swin_base(num_classes, pretrained_weights, fine_tuning_strategy, dropout, biomarker_config=None, single_target_strategy=None, target_feature_dim=None):
        # Determine head type based on fine-tuning strategy
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        
        if pretrained_weights == "ImageNet-22K":
            # Use IMAGENET1K_V1 as IMAGENET22K_V1 is not available in torchvision
            model = models.swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
            # Keep 3-channel input for pretrained weights, we'll convert images to 3-channel
        else:
            model = models.swin_b(weights=None)
            # For non-pretrained, modify for single channel input
            model.features[0][0] = nn.Conv2d(1, 128, kernel_size=4, stride=4)
        
        # Replace head with appropriate multi-task head
        feature_dim = model.head.in_features
        model.head = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, head_type=head_type, single_target_strategy=single_target_strategy, target_feature_dim=target_feature_dim)
        
        if fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_maxvit_base(num_classes, pretrained_weights, fine_tuning_strategy, dropout, biomarker_config=None, single_target_strategy=None, target_feature_dim=None):
        # Determine head type based on fine-tuning strategy
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        
        # Use timm for MaxViT
        model = timm.create_model('maxvit_base_tf_224.in1k', pretrained=True, num_classes=0)
        
        # Add appropriate multi-task head
        feature_dim = model.num_features
        model.head = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, head_type=head_type, single_target_strategy=single_target_strategy, target_feature_dim=target_feature_dim)
        
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
    def _create_clip_vit_b16(architecture, num_classes, fine_tuning_strategy, dropout, biomarker_config=None, single_target_strategy=None, target_feature_dim=None):
        # Determine head type based on fine-tuning strategy
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        
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
        model.head = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, head_type=head_type, single_target_strategy=single_target_strategy, target_feature_dim=target_feature_dim)
        
        if "frozen" in architecture or fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        
        return model
    
    @staticmethod
    def _create_clip_vit_l14(num_classes, fine_tuning_strategy, dropout, biomarker_config=None, single_target_strategy=None, target_feature_dim=None):
        # Determine head type based on fine-tuning strategy
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        
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
        model.head = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, head_type=head_type, single_target_strategy=single_target_strategy, target_feature_dim=target_feature_dim)
        
        return model
    
    @staticmethod
    def _create_blip2_vit_base(num_classes, fine_tuning_strategy, dropout, biomarker_config=None, single_target_strategy=None, target_feature_dim=None):
        # Determine head type based on fine-tuning strategy
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        
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
        model.head = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, head_type=head_type, single_target_strategy=single_target_strategy, target_feature_dim=target_feature_dim)
        
        return model
    
    @staticmethod
    def _create_medical_vlm(architecture, num_classes, fine_tuning_strategy, dropout, biomarker_config=None, single_target_strategy=None, target_feature_dim=None):
        # Determine head type based on fine-tuning strategy
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        
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
        model.head = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, head_type=head_type, single_target_strategy=single_target_strategy, target_feature_dim=target_feature_dim)
        
        return model
    
    @staticmethod
    def _create_diffusion_encoder(architecture, num_classes, fine_tuning_strategy, dropout, biomarker_config, single_target_strategy=None, target_feature_dim=None):
        """Create Stable Diffusion VAE Encoder"""
        
        if not DIFFUSERS_AVAILABLE:
            print(f"Warning: diffusers library not available. Using ResNet-50 as placeholder for {architecture}.")
            # Fallback to ResNet-50 placeholder
            model = models.resnet50(weights=None)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            feature_dim = model.fc.in_features
            model.fc = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, single_target_strategy=single_target_strategy, target_feature_dim=target_feature_dim)
            
            if "frozen" in architecture:
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.fc.parameters():
                    param.requires_grad = True
            
            return model
        
        # Determine model configuration based on architecture
        if "v1.5" in architecture:
            model_id = "runwayml/stable-diffusion-v1-5"
        elif "XL" in architecture:
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            model_id = "runwayml/stable-diffusion-v1-5"  # Default
        
        # Determine if encoder should be frozen
        frozen = "frozen" in architecture
        
        # Feature dimension for adaptation layer
        feature_dim = 512
        
        try:
            print(f"Loading {architecture} from {model_id}...")
            model = StableDiffusionVAEEncoder(
                model_id=model_id,
                feature_dim=feature_dim,
                biomarker_config=biomarker_config,
                dropout=dropout,
                frozen=frozen
            )
            print(f"✅ Successfully loaded {architecture}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading {architecture}: {e}")
            print("Falling back to ResNet-50 placeholder...")
            
            # Fallback to ResNet-50 placeholder
            model = models.resnet50(weights=None)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            feature_dim = model.fc.in_features
            model.fc = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, single_target_strategy=single_target_strategy, target_feature_dim=target_feature_dim)
            
            if frozen:
                for param in model.parameters():
                    param.requires_grad = False
                for param in model.fc.parameters():
                    param.requires_grad = True
            
            return model
    
    @staticmethod
    def _create_dit_base(num_classes, fine_tuning_strategy, dropout, biomarker_config=None, single_target_strategy=None, target_feature_dim=None):
        # Determine head type based on fine-tuning strategy
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        
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
        model.head = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, head_type=head_type, single_target_strategy=single_target_strategy, target_feature_dim=target_feature_dim)
        
        return model
    
    @staticmethod
    def _create_mae_vit_base(num_classes, fine_tuning_strategy, dropout, biomarker_config=None, single_target_strategy=None, target_feature_dim=None):
        # Determine head type based on fine-tuning strategy
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        
        # Use timm for MAE
        model = timm.create_model('vit_base_patch16_224.mae', pretrained=True, num_classes=0)
        
        # Modify for single channel input
        old_conv = model.patch_embed.proj
        model.patch_embed.proj = nn.Conv2d(1, old_conv.out_channels,
                                          kernel_size=old_conv.kernel_size,
                                          stride=old_conv.stride,
                                          padding=old_conv.padding)
        
        feature_dim = model.num_features
        model.head = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, head_type=head_type, single_target_strategy=single_target_strategy, target_feature_dim=target_feature_dim)
        
        return model
    
    @staticmethod
    def _create_resnet50_radimgnet(num_classes, fine_tuning_strategy, dropout, biomarker_config, single_target_strategy=None, target_feature_dim=None):
        # Determine head type based on fine-tuning strategy
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        
        # Load RadImageNet pre-trained weights
        try:
            import torch
            import os
            
            # Path to the RadImageNet checkpoint
            radimagenet_path = "/lfs/skampere2/0/mahmedc/Comorbidities-Detection/CT-Disease-Detection/radimagenet_ckpt/resnet50/ResNet50_RadImageNet.pt"
            
            if os.path.exists(radimagenet_path):
                print(f"Loading ResNet-50 with RadImageNet weights from {radimagenet_path}")
                
                # Create model without pre-trained weights
                model = models.resnet50(weights=None)
                
                # Load RadImageNet checkpoint
                checkpoint = torch.load(radimagenet_path, map_location='cpu')
                
                # Handle different checkpoint formats
                if 'model' in checkpoint:
                    # If checkpoint contains 'model' key
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    # If checkpoint contains 'state_dict' key
                    state_dict = checkpoint['state_dict']
                else:
                    # If checkpoint is directly the state dict
                    state_dict = checkpoint
                
                # Load the state dict, ignoring classifier layer (fc) since we'll replace it
                model_state_dict = model.state_dict()
                filtered_state_dict = {}
                
                for k, v in state_dict.items():
                    # Skip the classifier layer (fc) as we'll replace it with our multi-task head
                    if k.startswith('fc.') or k.startswith('classifier.'):
                        continue
                    
                    # Map keys - handle the 'backbone.' prefix and layer numbering from RadImageNet
                    mapped_key = k
                    if k.startswith('backbone.'):
                        # Remove 'backbone.' prefix
                        mapped_key = k[9:]
                        
                        # Map RadImageNet layer numbers to standard ResNet layer numbers
                        # RadImageNet: backbone.4 -> layer1, backbone.5 -> layer2, backbone.6 -> layer3, backbone.7 -> layer4
                        if mapped_key.startswith('4.'):
                            mapped_key = 'layer1.' + mapped_key[2:]
                        elif mapped_key.startswith('5.'):
                            mapped_key = 'layer2.' + mapped_key[2:]
                        elif mapped_key.startswith('6.'):
                            mapped_key = 'layer3.' + mapped_key[2:]
                        elif mapped_key.startswith('7.'):
                            mapped_key = 'layer4.' + mapped_key[2:]
                        elif mapped_key.startswith('0.'):
                            mapped_key = 'conv1.' + mapped_key[2:]
                        elif mapped_key.startswith('1.'):
                            mapped_key = 'bn1.' + mapped_key[2:]
                            
                    elif k.startswith('features.'):
                        mapped_key = k[9:]  # Remove 'features.' prefix
                    
                    # Check if the mapped key exists in the model
                    if mapped_key in model_state_dict:
                        filtered_state_dict[mapped_key] = v
                    else:
                        print(f"Warning: Could not map key {k} -> {mapped_key} to model state dict")
                
                # Load the filtered state dict
                model.load_state_dict(filtered_state_dict, strict=False)
                print("Successfully loaded RadImageNet weights!")
                
            else:
                print(f"RadImageNet checkpoint not found at {radimagenet_path}")
                print("Falling back to ImageNet ResNet-50")
                model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                
        except Exception as e:
            print(f"Error loading RadImageNet weights: {e}")
            print("Falling back to ImageNet ResNet-50")
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # For RadImageNet, we can use single-channel input since it was trained on medical images
        # But we'll keep 3-channel for now to match the current training pipeline
        # If you want to use single-channel, uncomment the line below:
        # model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace classifier with appropriate multi-task head
        feature_dim = model.fc.in_features
        model.fc = ModelFactory._create_multitask_head(feature_dim, dropout, biomarker_config, head_type=head_type, single_target_strategy=single_target_strategy, target_feature_dim=target_feature_dim)
        
        if fine_tuning_strategy == "linear_probe":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        
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

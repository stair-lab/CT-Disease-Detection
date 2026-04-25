"""
Single Target Strategy Implementations
Handles different feature extraction strategies for single-target classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import Union, Tuple, Optional


class SingleTargetStrategy(Enum):
    """Enumeration of single-target classification strategies"""
    DIRECT_CLASSIFICATION_HEAD = "Direct classification head"
    CLS_TOKEN_CLASSIFICATION = "CLS token classification"
    GLOBAL_AVERAGE_POOLING = "Global average pooling"


class FeatureExtractor(nn.Module):
    """Base class for feature extraction strategies"""
    
    def __init__(self, strategy: SingleTargetStrategy):
        super().__init__()
        self.strategy = strategy
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features based on the strategy"""
        raise NotImplementedError


class DirectClassificationHeadExtractor(FeatureExtractor):
    """
    Direct classification head strategy for CNN-based models
    Uses global average pooling followed by classification head
    """
    
    def __init__(self, input_dim: int, feature_dim: int = 512, dropout: float = 0.1):
        super().__init__(SingleTargetStrategy.DIRECT_CLASSIFICATION_HEAD)
        
        # Global average pooling for spatial features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(feature_dim)
        )
        
        self.output_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using direct classification head approach
        
        Args:
            x: Input features [B, C, H, W] (spatial features from CNN) or [B, input_dim] (flattened features)
            
        Returns:
            Processed features [B, feature_dim]
        """
        # Check if input is spatial (4D) or flattened (2D)
        if x.dim() == 4:
            # Spatial features: apply global average pooling first
            pooled = self.global_pool(x)  # [B, C, 1, 1]
            features = self.feature_processor(pooled)  # [B, feature_dim]
        elif x.dim() == 2:
            # Flattened features: skip global pooling, go directly to feature processing
            # But we need to adjust the input dimension for the linear layer
            # The input_dim in the constructor was for spatial features, but now we have flattened features
            # We need to create a new feature processor for the flattened input
            if not hasattr(self, 'flattened_processor'):
                # Create a new processor for flattened features
                self.flattened_processor = nn.Sequential(
                    nn.Linear(x.size(1), self.output_dim),  # x.size(1) is the actual flattened dimension
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.feature_processor[3].p),  # Use same dropout rate
                    nn.LayerNorm(self.output_dim)  # Use LayerNorm instead of BatchNorm to avoid single-sample issues
                )
                # Move to the same device as the input tensor
                self.flattened_processor = self.flattened_processor.to(x.device)
            features = self.flattened_processor(x)  # [B, feature_dim]
        else:
            raise ValueError(f"Expected 2D or 4D input tensor, got {x.dim()}D tensor with shape {x.shape}")
        
        return features


class CLSTokenClassificationExtractor(FeatureExtractor):
    """
    CLS token classification strategy for Transformer-based models
    Extracts the CLS token from transformer output
    """
    
    def __init__(self, feature_dim: int = 768, dropout: float = 0.1):
        super().__init__(SingleTargetStrategy.CLS_TOKEN_CLASSIFICATION)
        
        # Feature processing for CLS token
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(feature_dim)
        )
        
        self.output_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS token features from transformer output
        
        Args:
            x: Transformer output [B, seq_len, feature_dim] or [B, feature_dim]
            
        Returns:
            CLS token features [B, feature_dim]
        """
        # Handle different input shapes
        if x.dim() == 3:  # [B, seq_len, feature_dim]
            # Extract CLS token (first token)
            cls_token = x[:, 0, :]  # [B, feature_dim]
        elif x.dim() == 2:  # [B, feature_dim] - already extracted
            cls_token = x
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Process CLS token features
        features = self.feature_processor(cls_token)  # [B, feature_dim]
        
        return features


class GlobalAveragePoolingExtractor(FeatureExtractor):
    """
    Global average pooling strategy for models with spatial feature maps
    Used for VAE encoders and other models that output spatial features
    """
    
    def __init__(self, input_dim: int, feature_dim: int = 512, dropout: float = 0.1):
        super().__init__(SingleTargetStrategy.GLOBAL_AVERAGE_POOLING)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(feature_dim)
        )
        
        self.output_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using global average pooling
        
        Args:
            x: Input features [B, C, H, W] (spatial features)
            
        Returns:
            Processed features [B, feature_dim]
        """
        # Apply global average pooling
        pooled = self.global_pool(x)  # [B, C, 1, 1]
        
        # Process through feature layers
        features = self.feature_processor(pooled)  # [B, feature_dim]
        
        return features


def create_feature_extractor(
    strategy: Union[str, SingleTargetStrategy], 
    input_dim: int, 
    feature_dim: int = 512, 
    dropout: float = 0.1
) -> FeatureExtractor:
    """
    Factory function to create appropriate feature extractor
    
    Args:
        strategy: Single-target strategy (string or enum)
        input_dim: Input feature dimension
        feature_dim: Output feature dimension
        dropout: Dropout rate
        
    Returns:
        Appropriate FeatureExtractor instance
    """
    # Convert string to enum if needed
    if isinstance(strategy, str):
        strategy = SingleTargetStrategy(strategy)
    
    if strategy == SingleTargetStrategy.DIRECT_CLASSIFICATION_HEAD:
        return DirectClassificationHeadExtractor(input_dim, feature_dim, dropout)
    
    elif strategy == SingleTargetStrategy.CLS_TOKEN_CLASSIFICATION:
        return CLSTokenClassificationExtractor(feature_dim, dropout)
    
    elif strategy == SingleTargetStrategy.GLOBAL_AVERAGE_POOLING:
        return GlobalAveragePoolingExtractor(input_dim, feature_dim, dropout)
    
    else:
        raise ValueError(f"Unknown single-target strategy: {strategy}")


def extract_features_from_model_output(
    model_output: torch.Tensor, 
    strategy: Union[str, SingleTargetStrategy],
    input_dim: Optional[int] = None,
    feature_dim: int = 512,
    dropout: float = 0.1
) -> torch.Tensor:
    """
    Extract features from model output based on strategy
    
    Args:
        model_output: Raw output from backbone model
        strategy: Single-target strategy to use
        input_dim: Input dimension (required for some strategies)
        feature_dim: Output feature dimension
        dropout: Dropout rate
        
    Returns:
        Extracted features [B, feature_dim]
    """
    # Convert string to enum if needed
    if isinstance(strategy, str):
        strategy = SingleTargetStrategy(strategy)
    
    if strategy == SingleTargetStrategy.DIRECT_CLASSIFICATION_HEAD:
        if input_dim is None:
            raise ValueError("input_dim required for DIRECT_CLASSIFICATION_HEAD strategy")
        extractor = DirectClassificationHeadExtractor(input_dim, feature_dim, dropout)
        return extractor(model_output)
    
    elif strategy == SingleTargetStrategy.CLS_TOKEN_CLASSIFICATION:
        extractor = CLSTokenClassificationExtractor(feature_dim, dropout)
        return extractor(model_output)
    
    elif strategy == SingleTargetStrategy.GLOBAL_AVERAGE_POOLING:
        if input_dim is None:
            raise ValueError("input_dim required for GLOBAL_AVERAGE_POOLING strategy")
        extractor = GlobalAveragePoolingExtractor(input_dim, feature_dim, dropout)
        return extractor(model_output)
    
    else:
        raise ValueError(f"Unknown single-target strategy: {strategy}")


# Strategy mapping from config string values to enum values
STRATEGY_MAPPING = {
    "Direct classification head": SingleTargetStrategy.DIRECT_CLASSIFICATION_HEAD,
    "CLS token classification": SingleTargetStrategy.CLS_TOKEN_CLASSIFICATION,
    "Global average pooling": SingleTargetStrategy.GLOBAL_AVERAGE_POOLING,
}


def get_strategy_from_name(strategy_name: str) -> SingleTargetStrategy:
    """
    Convert strategy string value to SingleTargetStrategy enum.
    
    Args:
        strategy_name: Strategy string from config/checkpoint
        
    Returns:
        SingleTargetStrategy enum value
    """
    if strategy_name not in STRATEGY_MAPPING:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Available: {list(STRATEGY_MAPPING.keys())}"
        )
    
    return STRATEGY_MAPPING[strategy_name]


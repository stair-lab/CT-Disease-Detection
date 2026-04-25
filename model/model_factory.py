"""
Model Factory for architectures used in this release.
"""

from pathlib import Path

import timm
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    Swin_B_Weights,
)
from config.experiment_config import MODEL_DEFAULTS



class MultiTaskHead(nn.Module):
    """Legacy fallback head used when biomarker_config is not provided."""

    def __init__(self, input_dim, num_binary_tasks=7, num_calcium_classes=4, num_regression_tasks=2, dropout=0.1):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(512),
        )
        self.binary_head = nn.Linear(512, num_binary_tasks)
        self.calcium_head = nn.Linear(512, num_calcium_classes)
        self.regression_head = nn.Linear(512, num_regression_tasks)

    def forward(self, x):
        shared = self.shared_layers(x)
        binary_out = self.binary_head(shared)
        calcium_out = self.calcium_head(shared)
        regression_out = self.regression_head(shared)
        return torch.cat([binary_out, calcium_out, regression_out], dim=1)


class ModelFactory:
    """Factory class for supported public model architectures."""

    SUPPORTED_ARCHITECTURES = tuple(MODEL_DEFAULTS.keys())

    @staticmethod
    def create_model(
        architecture,
        num_classes=13,
        pretrained_weights=None,
        fine_tuning_strategy="full",
        dropout=0.1,
        biomarker_config=None,
        single_target_strategy=None,
        target_feature_dim=None,
        single_target_output_dim=None,
        **kwargs,
    ):
        """Create model based on architecture specification."""
        if single_target_output_dim is not None:
            target_feature_dim = single_target_output_dim

        if architecture == "ResNet-18":
            return ModelFactory._create_resnet18(
                num_classes,
                pretrained_weights,
                fine_tuning_strategy,
                dropout,
                biomarker_config,
                single_target_strategy,
                target_feature_dim,
            )
        if architecture == "ResNet-34":
            return ModelFactory._create_resnet34(
                num_classes,
                pretrained_weights,
                fine_tuning_strategy,
                dropout,
                biomarker_config,
                single_target_strategy,
                target_feature_dim,
            )
        if architecture == "DenseNet-121":
            return ModelFactory._create_densenet121(
                num_classes,
                pretrained_weights,
                fine_tuning_strategy,
                dropout,
                biomarker_config,
                single_target_strategy,
                target_feature_dim,
            )
        if architecture == "EfficientNet-B0":
            return ModelFactory._create_efficientnet_b0(
                num_classes,
                pretrained_weights,
                fine_tuning_strategy,
                dropout,
                biomarker_config,
                single_target_strategy,
                target_feature_dim,
            )
        if architecture == "ViT-Small (DINOv2)":
            return ModelFactory._create_dinov2_vit(
                architecture,
                num_classes,
                fine_tuning_strategy,
                dropout,
                biomarker_config,
                single_target_strategy,
                target_feature_dim,
            )
        if architecture == "Swin Transformer-Base":
            return ModelFactory._create_swin_base(
                num_classes,
                pretrained_weights,
                fine_tuning_strategy,
                dropout,
                biomarker_config,
                single_target_strategy,
                target_feature_dim,
            )
        if architecture == "ResNet-50 (RadImageNet)":
            return ModelFactory._create_resnet50_radimgnet(
                num_classes,
                fine_tuning_strategy,
                dropout,
                biomarker_config,
                single_target_strategy,
                target_feature_dim,
            )

        raise ValueError(
            f"Unsupported architecture: {architecture}. "
            f"Supported: {list(ModelFactory.SUPPORTED_ARCHITECTURES)}"
        )

    @staticmethod
    def _create_multitask_head(feature_dim, dropout, biomarker_config, head_type="flexible", single_target_strategy=None, target_feature_dim=None):
        if biomarker_config is not None:
            if head_type == "linear_probe":
                from .flexible_multitask_head import LinearProbeMultiTaskHead

                return LinearProbeMultiTaskHead(
                    feature_dim,
                    biomarker_config,
                    dropout=dropout,
                    single_target_strategy=single_target_strategy,
                    target_feature_dim=target_feature_dim,
                )
            from .flexible_multitask_head import FlexibleMultiTaskHead

            return FlexibleMultiTaskHead(
                feature_dim,
                biomarker_config,
                dropout=dropout,
                single_target_strategy=single_target_strategy,
                target_feature_dim=target_feature_dim,
            )
        return MultiTaskHead(feature_dim, dropout=dropout)

    @staticmethod
    def _freeze_for_linear_probe(model, head_module):
        for param in model.parameters():
            param.requires_grad = False
        for param in head_module.parameters():
            param.requires_grad = True

    @staticmethod
    def _create_resnet18(num_classes, pretrained_weights, fine_tuning_strategy, dropout, biomarker_config, single_target_strategy=None, target_feature_dim=None):
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        if pretrained_weights == "ImageNet":
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        feature_dim = model.fc.in_features
        model.fc = ModelFactory._create_multitask_head(
            feature_dim, dropout, biomarker_config, head_type, single_target_strategy, target_feature_dim
        )
        if fine_tuning_strategy == "linear_probe":
            ModelFactory._freeze_for_linear_probe(model, model.fc)
        return model

    @staticmethod
    def _create_resnet34(num_classes, pretrained_weights, fine_tuning_strategy, dropout, biomarker_config, single_target_strategy=None, target_feature_dim=None):
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        if pretrained_weights == "ImageNet":
            model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet34(weights=None)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        feature_dim = model.fc.in_features
        model.fc = ModelFactory._create_multitask_head(
            feature_dim, dropout, biomarker_config, head_type, single_target_strategy, target_feature_dim
        )
        if fine_tuning_strategy == "linear_probe":
            ModelFactory._freeze_for_linear_probe(model, model.fc)
        return model

    @staticmethod
    def _create_densenet121(num_classes, pretrained_weights, fine_tuning_strategy, dropout, biomarker_config=None, single_target_strategy=None, target_feature_dim=None):
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        if pretrained_weights == "ImageNet":
            model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            model = models.densenet121(weights=None)
        feature_dim = model.classifier.in_features
        model.classifier = ModelFactory._create_multitask_head(
            feature_dim,
            dropout,
            biomarker_config,
            head_type=head_type,
            single_target_strategy=single_target_strategy,
            target_feature_dim=target_feature_dim,
        )
        if fine_tuning_strategy == "linear_probe":
            ModelFactory._freeze_for_linear_probe(model, model.classifier)
        return model

    @staticmethod
    def _create_efficientnet_b0(num_classes, pretrained_weights, fine_tuning_strategy, dropout, biomarker_config=None, single_target_strategy=None, target_feature_dim=None):
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        if pretrained_weights == "ImageNet":
            model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b0(weights=None)
        feature_dim = model.classifier[1].in_features
        model.classifier = ModelFactory._create_multitask_head(
            feature_dim,
            dropout,
            biomarker_config,
            head_type=head_type,
            single_target_strategy=single_target_strategy,
            target_feature_dim=target_feature_dim,
        )
        if fine_tuning_strategy == "linear_probe":
            ModelFactory._freeze_for_linear_probe(model, model.classifier)
        return model

    @staticmethod
    def _create_dinov2_vit(architecture, num_classes, fine_tuning_strategy, dropout, biomarker_config, single_target_strategy=None, target_feature_dim=None):
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        model = timm.create_model("vit_small_patch14_dinov2", pretrained=True, num_classes=0, img_size=256)
        feature_dim = model.num_features
        model.head = ModelFactory._create_multitask_head(
            feature_dim,
            dropout,
            biomarker_config,
            head_type=head_type,
            single_target_strategy=single_target_strategy,
            target_feature_dim=target_feature_dim,
        )
        if fine_tuning_strategy == "linear_probe":
            ModelFactory._freeze_for_linear_probe(model, model.head)
        return model

    @staticmethod
    def _create_swin_base(num_classes, pretrained_weights, fine_tuning_strategy, dropout, biomarker_config=None, single_target_strategy=None, target_feature_dim=None):
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        if pretrained_weights == "ImageNet-22K":
            model = models.swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        else:
            model = models.swin_b(weights=None)
            model.features[0][0] = nn.Conv2d(1, 128, kernel_size=4, stride=4)
        feature_dim = model.head.in_features
        model.head = ModelFactory._create_multitask_head(
            feature_dim,
            dropout,
            biomarker_config,
            head_type=head_type,
            single_target_strategy=single_target_strategy,
            target_feature_dim=target_feature_dim,
        )
        if fine_tuning_strategy == "linear_probe":
            ModelFactory._freeze_for_linear_probe(model, model.head)
        return model

    @staticmethod
    def _create_resnet50_radimgnet(num_classes, fine_tuning_strategy, dropout, biomarker_config, single_target_strategy=None, target_feature_dim=None):
        head_type = "linear_probe" if fine_tuning_strategy == "linear_probe" else "flexible"
        ckpt_path = (
            Path(__file__).resolve().parents[1]
            / "radimagenet_ckpt"
            / "resnet50"
            / "ResNet50_RadImageNet.pt"
        )
        try:
            if ckpt_path.exists():
                model = models.resnet50(weights=None)
                checkpoint = torch.load(str(ckpt_path), map_location="cpu")
                state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
                model_state_dict = model.state_dict()
                filtered_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("fc.") or key.startswith("classifier."):
                        continue
                    mapped_key = key
                    if key.startswith("backbone."):
                        mapped_key = key[9:]
                        if mapped_key.startswith("4."):
                            mapped_key = "layer1." + mapped_key[2:]
                        elif mapped_key.startswith("5."):
                            mapped_key = "layer2." + mapped_key[2:]
                        elif mapped_key.startswith("6."):
                            mapped_key = "layer3." + mapped_key[2:]
                        elif mapped_key.startswith("7."):
                            mapped_key = "layer4." + mapped_key[2:]
                        elif mapped_key.startswith("0."):
                            mapped_key = "conv1." + mapped_key[2:]
                        elif mapped_key.startswith("1."):
                            mapped_key = "bn1." + mapped_key[2:]
                    elif key.startswith("features."):
                        mapped_key = key[9:]
                    if mapped_key in model_state_dict:
                        filtered_state_dict[mapped_key] = value
                model.load_state_dict(filtered_state_dict, strict=False)
            else:
                model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        feature_dim = model.fc.in_features
        model.fc = ModelFactory._create_multitask_head(
            feature_dim,
            dropout,
            biomarker_config,
            head_type=head_type,
            single_target_strategy=single_target_strategy,
            target_feature_dim=target_feature_dim,
        )
        if fine_tuning_strategy == "linear_probe":
            ModelFactory._freeze_for_linear_probe(model, model.fc)
        return model


def get_model_memory_requirement(architecture):
    """Get expected GPU memory requirement for supported public architectures."""
    memory_map = {
        "ResNet-18": "4-6GB",
        "ResNet-34": "6-8GB",
        "DenseNet-121": "8-10GB",
        "EfficientNet-B0": "6-8GB",
        "ViT-Small (DINOv2)": "8-10GB",
        "Swin Transformer-Base": "12-16GB",
        "ResNet-50 (RadImageNet)": "6-8GB",
    }
    return memory_map.get(architecture, "Unknown")

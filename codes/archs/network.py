import torch
import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    resnet18, ResNet18_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    regnet_x_400mf, RegNet_X_400MF_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights
)

# ======================= UTILITIES =======================

def replace_first_conv(model, new_in_channels):
    """
    Find and replace the first Conv2d layer to accept new_in_channels.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"✅ Replacing first Conv2d layer at: {name}")
            # Create a new Conv2d layer
            new_conv = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode
            )
            # Copy existing pretrained weights smartly
            with torch.no_grad():
                if new_in_channels <= module.in_channels:
                    new_conv.weight[:, :new_in_channels] = module.weight[:, :new_in_channels]
                else:
                    new_conv.weight[:, :module.in_channels] = module.weight
            if module.bias is not None:
                new_conv.bias = module.bias

            # Assign the new conv layer
            parent_name = ".".join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]
            parent_module = model if parent_name == "" else dict(model.named_modules())[parent_name]
            setattr(parent_module, attr_name, new_conv)
            return
    raise ValueError("No Conv2d layer found to replace input channels.")

def find_classifier(model):
    """
    Find the final classifier layer (Linear) to replace it for c_out outputs.
    """
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        return model, 'fc'
    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            for idx in reversed(range(len(model.classifier))):
                if isinstance(model.classifier[idx], nn.Linear):
                    return model.classifier, str(idx)
        elif isinstance(model.classifier, nn.Linear):
            return model, 'classifier'
    raise NotImplementedError("Custom output head not implemented for this model.")

# ======================= MODEL LOADER =======================

def load_model(model_arch_name, channel, c_out=1, pretrained=True):
    """
    Load model, modify input and output layers according to channel and c_out.
    """
    if model_arch_name == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        out_dim = 2048
    elif model_arch_name == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        out_dim = 512
    elif model_arch_name == 'convnext_tiny':
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        out_dim = 768
    elif model_arch_name == 'regnet_x_400mf':
        weights = RegNet_X_400MF_Weights.IMAGENET1K_V2 if pretrained else None
        model = regnet_x_400mf(weights=weights)
        out_dim = 400
    elif model_arch_name == 'mobilenet_v3_small':
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        out_dim = 576
    else:
        raise ValueError(f"Unknown model architecture: {model_arch_name}")

    # Adjust input layer
    replace_first_conv(model, channel)

    # Adjust output layer
    classifier_module, classifier_attr = find_classifier(model)
    old_classifier = getattr(classifier_module, classifier_attr)
    new_classifier = nn.Linear(old_classifier.in_features, c_out)
    setattr(classifier_module, classifier_attr, new_classifier)

    return model, out_dim

# ======================= SINGLEMODAL NETWORK =======================

class SinglemodalNet(nn.Module):
    def __init__(self, model_arch_name, channel, pretrained=True):
        super().__init__()
        self.model, _ = load_model(model_arch_name, channel, 1, pretrained)

    def forward(self, x):
        return self.model(x)

# ======================= MULTIMODAL NETWORK =======================

class MultimodalNet(nn.Module):
    def __init__(self, model_arch_name, channel, fusion_mode='E', pretrained=True):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.model_arch_name = model_arch_name

        # For late fusion: Split into two different channel groups
        c1, c2 = 3, channel - 3

        if fusion_mode == 'E':
            self.model, _ = load_model(model_arch_name, channel, 1, pretrained)

        elif fusion_mode == 'L':
            model1, out_dim = load_model(model_arch_name, c1, 1, pretrained)
            model2, _ = load_model(model_arch_name, c2, 1, pretrained)

            # Assuming standard model blocks
            self.model1 = nn.Sequential(
                model1.conv1, model1.bn1, model1.relu, model1.maxpool,
                model1.layer1, model1.layer2, model1.layer3, model1.layer4
            )
            self.model2 = nn.Sequential(
                model2.conv1, model2.bn1, model2.relu, model2.maxpool,
                model2.layer1, model2.layer2, model2.layer3, model2.layer4
            )

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Sequential(
                nn.Linear(out_dim * 2, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")

    def forward(self, x):
        if self.fusion_mode == 'E':
            x1, x2 = x
            x = torch.cat([x1, x2], dim=1)
            return self.model(x)
        
        elif self.fusion_mode == 'L':
            x1, x2 = x
            x1 = self.model1(x1)
            x2 = self.model2(x2)
            x1 = self.avgpool(x1).flatten(start_dim=1)
            x2 = self.avgpool(x2).flatten(start_dim=1)
            x_fusion = torch.cat([x1, x2], dim=1)
            return self.fc(x_fusion)

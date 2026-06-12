import torch
import torch.nn as nn
from torchvision import models

class PalmVeinNet(nn.Module):
    def __init__(self, num_classes=2):
        """
        Uses an EfficientNet-B0 backbone for high accuracy 
        biometric recognition with a low memory footprint.
        """
        super(PalmVeinNet, self).__init__()
        # Load weights safely using the modern PyTorch weights API
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Replace the final classification head to match our dataset's classes
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def get_transforms():
    """
    Standard pre-processing transformations for the images 
    to match the Expected ImageNet input sizes.
    """
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
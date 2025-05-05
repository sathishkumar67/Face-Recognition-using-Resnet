from __future__ import annotations
import torch.nn as nn
import torchvision


class SiameseResNet(nn.Module):
    def __init__(self, embedding_dim=256):
        super(SiameseResNet, self).__init__()
        # Load pretrained ResNet18
        self.backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1", progress=True)
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Linear(512, embedding_dim)  # 512 -> 256

    def forward(self, x):
        return self.backbone(x)
    
    def print_parameters_count(self):
        total_params = sum(p.numel() for p in self.parameters()) / 1e6  # Convert to millions
        # Print the number of parameters in millions
        print(f"Total parameters: {total_params:.2f}m")
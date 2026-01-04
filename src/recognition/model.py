# src/recognition/model.py

import torch
import torch.nn as nn
import torchvision.models as models

class SketchANet(nn.Module):
    """使用 ResNet18 作为 backbone 的轻量分类模型"""
    def __init__(self, num_classes=5):
        super(SketchANet, self).__init__()
        backbone = models.resnet18(weights=None)
        self.features = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),  
            *list(backbone.children())[:-1]  
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
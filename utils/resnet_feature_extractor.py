import torch
import timm
import torch.nn as nn

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, model_name="resnet50"):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)

    def forward(self, x):
        return self.model(x)

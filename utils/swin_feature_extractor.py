import timm
import torch
import torch.nn as nn

class SwinFeatureExtractor(nn.Module):
    def __init__(self, model_name="swin_tiny_patch4_window7_224"):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)

    def forward(self, x):
        return self.model(x)

import timm
import torch
import torch.nn as nn
from timm.models.vision_transformer import vit_base_patch16_224
from torchvision import models

def get_model(name="vgg16", num_classes=4):
    if name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif name == "alexnet":
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif name == "squeezenet":
        model = models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
        model.num_classes = num_classes

    else:
        raise ValueError("Nieznany model")

    # Zamrażamy featury
    for param in model.parameters():
        param.requires_grad = False

    # Odmrażamy klasyfikator
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def get_retfound_mae(num_classes=4, checkpoint_path=None):
    model = vit_base_patch16_224(pretrained=False)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)

    # Podmieniamy classification head
    model.head = nn.Linear(model.head.in_features, num_classes)

    return model


def get_medvit(num_classes=4, checkpoint_path=None):
    model = timm.create_model('medvit_base_patch16_224', pretrained=False, num_classes=num_classes)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    if hasattr(model, 'head'):
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif hasattr(model, 'classifier'):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model

def get_swin(num_classes=4, pretrained=True):
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)
    if hasattr(model, 'head'):
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif hasattr(model, 'classifier'):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

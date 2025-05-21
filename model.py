import torch.nn as nn
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

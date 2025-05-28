import torch
import timm
from timm import create_model

model = create_model(
        args.model,
        num_classes=args.nb_classes,
    )
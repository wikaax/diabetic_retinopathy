import matplotlib.pyplot as plt
import pandas as pd
import torch

import os

base_path = "data/train"

for class_folder in sorted(os.listdir(base_path)):
    class_path = os.path.join(base_path, class_folder)
    if os.path.isdir(class_path):
        count = len([
            fname for fname in os.listdir(class_path)
            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        print(f"{class_folder}: {count} obrazów")


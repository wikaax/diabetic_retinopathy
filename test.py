import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from dataloader import get_loaders
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "vgg16"
num_classes = 4
batch_size = 32

df = pd.read_csv("messidor_data_modified.csv")
train_dataset_labels = df['diagnosis'].astype(int).values
classes = np.array(sorted(np.unique(train_dataset_labels)))

_, val_loader = get_loaders("messidor_data_modified.csv", "data/processed", batch_size=batch_size)

model = get_model(model_name, num_classes=num_classes)
model = model.to(device)

checkpoint_path = f"./checkpoints/{model_name}_best.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Classification Report:")
print(classification_report(all_labels, all_preds))
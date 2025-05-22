import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import compute_class_weight
import seaborn as sns
import os

from model import get_model, get_retfound_mae, get_medvit, get_swin
from dataloader import get_loaders
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("messidor_data.csv")
df['diagnosis'] = df['diagnosis'].replace({4: 3})
df.to_csv("messidor_data_modified.csv", index=False)

train_dataset_labels = df['diagnosis'].astype(int).values
classes = np.array(sorted(np.unique(train_dataset_labels)))

class_weights = compute_class_weight('balanced', classes=classes, y=train_dataset_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

model_name = "vgg16"
model = get_model(model_name, num_classes=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_loader, val_loader = get_loaders("messidor_data_modified.csv", "data/processed", batch_size=32)

num_epochs = 20
best_val_loss = float('inf')
save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"{model_name}_best.pth")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print("Class weights:", class_weights)

    loop = tqdm(train_loader, desc=f"Epoka {epoch+1}/{num_epochs}", unit="batch")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        # outputs = outputs.permute(0, 3, 1, 2)  # (batch, classes, H, W)
        # outputs = outputs.mean(dim=[2, 3])  # średnia po H i W
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(train_loader)

    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total

    # cm = confusion_matrix(all_labels, all_preds)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.title("Confusion Matrix")
    # plt.show()

    print("\n Classification Report:")
    print(classification_report(all_labels, all_preds))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'accuracy': accuracy
        }, save_path)
        print(f"Save best model, val loss: {val_loss:.4f}")

    print(f"\nEpoka {epoch+1} zakończona, Średni loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_epoch{epoch + 1}.pth"))

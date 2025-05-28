# import os
# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# from PIL import Image
# import timm
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# def prepare_model(arch='swin_base_patch4_window7_224', pretrained=True):
#     model = timm.create_model(arch, pretrained=pretrained)
#     model.head = nn.Identity()
#     model.to(device).eval()
#     return model
#
# def run_one_image(img, model):
#     x = torch.tensor(img).unsqueeze(0)  # batch dim
#     x = torch.einsum('nhwc->nchw', x).float().to(device)
#     with torch.no_grad():
#         features = model.forward_features(x)
#
#     # global average pooling over spatial dimensions
#     features = features.mean(dim=[1, 2])
#
#     features = features.squeeze(0)  # usuwamy batch dim, teraz [1024]
#     return features
# def get_feature_from_folders(base_path, model):
#     name_list = []
#     feature_list = []
#
#     classes = os.listdir(base_path)
#     for cls in classes:
#         class_path = os.path.join(base_path, cls)
#         if not os.path.isdir(class_path):
#             continue
#         img_list = os.listdir(class_path)
#         for i, img_name in enumerate(img_list):
#             if i % 100 == 0:
#                 print(f"{i} images processed in class {cls}...")
#
#             img_path = os.path.join(class_path, img_name)
#             try:
#                 img = Image.open(img_path).convert("RGB")
#             except:
#                 print(f"Failed to open image: {img_path}")
#                 continue
#
#             img = img.resize((224, 224))
#             img = np.array(img) / 255.
#             for c in range(3):
#                 img[..., c] = (img[..., c] - img[..., c].mean()) / (img[..., c].std() + 1e-8)
#
#             if img.shape != (224, 224, 3):
#                 print(f"Wrong shape for {img_path}")
#                 continue
#
#             latent_feature = run_one_image(img, model)
#             name_list.append(f"{cls}/{img_name}")
#             feature_list.append(latent_feature.detach().cpu().numpy())
#
#     return name_list, feature_list
#
# if __name__ == '__main__':
#     data_path = 'data/train'  # ścieżka do zdjęć
#     model = prepare_model(arch='swin_base_patch4_window7_224', pretrained=True)
#
#     print("==> Extracting features from Swin Transformer...")
#     name_list, feature_list = get_feature_from_folders(data_path, model)
#
#     print("==> Saving features to CSV...")
#     df_feature = pd.DataFrame(feature_list)
#     df_imgname = pd.DataFrame(name_list, columns=["name"])
#     df_visualization = pd.concat([df_imgname, df_feature], axis=1)
#
#     column_names = ["feature_{}".format(i) for i in range(df_feature.shape[1])]
#     df_visualization.columns = ["name"] + column_names
#     df_visualization.to_csv("Swin_Features.csv", index=False)
#     print("Done. Features saved to Swin_Features.csv")


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder('data/train', transform=transform_train)
val_dataset = datasets.ImageFolder('data/val', transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

model = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=5)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def train_model(num_epochs=10):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")

        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        print(f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

        # Zapisywanie najlepszego modelu
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_swin_model.pth')
            print("💾 Best model saved!")

if __name__ == '__main__':
    train_model(num_epochs=10)

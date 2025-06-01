import numpy as np
import torch
import timm
from timm import create_model
from tqdm import tqdm

from models.MedVit.CustomDataset.MedViT import MedViT_large
from torchvision import transforms, datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MedViT_large(pretrained=True)
model.eval()
model.to(device)

def extract_features(model, x):
    x = model.stem(x)
    for layer in model.features:
        x = layer(x)
    x = model.norm(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # upewnij się, że MedViT tego wymaga
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # albo [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] jeśli ImageNet
])

dataset = datasets.ImageFolder('./data/train', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

all_embeddings = []


with torch.no_grad():
    for images, _ in tqdm(dataloader):
        images = images.to(device)
        emb = extract_features(model, images)
        all_embeddings.append(emb.cpu().numpy())

all_embeddings = np.concatenate(all_embeddings, axis=0)
np.save("medvit_embeddings.npy", all_embeddings)
print("✅ Saved to medvit_embeddings.npy")
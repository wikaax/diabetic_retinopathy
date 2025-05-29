import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from models.RETFound.models_vit import RETFound_mae
from utils.resnet_feature_extractor import ResNetFeatureExtractor
from utils.swin_feature_extractor import SwinFeatureExtractor
import TransRate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":


    # === Prepare data ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder("data/train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # === Load model ===
    model = RETFound_mae()
    model.to(device)
    model.eval()

    # === Feature extraction ===
    features = []
    labels = []

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.to(device)
            feats = model(x)
            features.append(feats.cpu().numpy())
            labels.append(y.cpu().numpy())

    Z = np.vstack(features)
    y = np.hstack(labels)

    # === Save if you want ===
    np.save("logs_trans/retfound_features.npy", Z)

    # === Calculate TransRate ===
    score = TransRate.transrate(Z, y, eps=1e-2)
    print(f"\n🔥 TransRate dla ResNet: {score:.4f}")
    # data = np.load("logs_trans/swin_tiny.npy")
    # print("Shape:", data.shape)
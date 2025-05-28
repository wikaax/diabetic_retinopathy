import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import external.RETFound_MAE.models_vit as models
from huggingface_hub import hf_hub_download

# Ustawienia
np.set_printoptions(threshold=np.inf)
np.random.seed(1)
torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_model(chkpt_dir, arch='RETFound_mae'):
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    if arch == 'vit_large_patch16':
        model = models.__dict__[arch](
            img_size=224,
            num_classes=5,
            drop_path_rate=0,
            global_pool=True,
        )
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model = models.__dict__[arch](
            num_classes=5,
            drop_path_rate=0,
            # args=None,
        )
        state_dict_key = 'model' if 'model' in checkpoint else 'teacher'
        model.load_state_dict(checkpoint[state_dict_key], strict=False)
    return model

def run_one_image(img, model, arch):
    x = torch.tensor(img).unsqueeze(dim=0)  # dodaje batch dimension
    x = torch.einsum('nhwc->nchw', x)  # zamiana kanałów na NCHW
    x = x.to(device, non_blocking=True)
    latent = model.forward_features(x.float())



    if arch == 'dinov2_large':
        latent = latent[:, 1:, :].mean(dim=1, keepdim=True)
        latent = nn.LayerNorm(latent.shape[-1], eps=1e-6).to(device)(latent)

    latent = torch.squeeze(latent)
    return latent

def get_feature_from_folders(base_path, chkpt_dir, arch='vit_large_patch16'):
    model_ = prepare_model(chkpt_dir, arch)
    model_.to(device)
    model_.eval()

    name_list = []
    feature_list = []

    classes = os.listdir(base_path)
    for cls in classes:
        class_path = os.path.join(base_path, cls)
        if not os.path.isdir(class_path):
            continue
        img_list = os.listdir(class_path)
        for i, img_name in enumerate(img_list):
            if i % 100 == 0:
                print(f"{i} images processed in class {cls}...")

            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
            except:
                print(f"Failed to open image: {img_path}")
                continue

            img = img.resize((224, 224))
            img = np.array(img) / 255.
            for c in range(3):
                img[..., c] = (img[..., c] - img[..., c].mean()) / (img[..., c].std() + 1e-8)

            if img.shape != (224, 224, 3):
                print(f"Wrong shape for {img_path}")
                continue

            latent_feature = run_one_image(img, model_, arch)
            name_list.append(f"{cls}/{img_name}")  # dla jasności dajemy info z klasą w nazwie
            feature_list.append(latent_feature.detach().cpu().numpy())

    return name_list, feature_list

def load_images_from_folders(base_path):
    images = []
    labels = []
    classes = os.listdir(base_path)
    for cls in classes:
        class_path = os.path.join(base_path, cls)
        if not os.path.isdir(class_path):
            continue  # pomijamy pliki, jeśli takie są
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            try:
                img = Image.open(fpath).convert('RGB')
                images.append(img)
                labels.append(cls)
            except Exception as e:
                print(f"Failed to open image: {fpath} ({e})")
    return images, labels

# === START HERE ===
if __name__ == '__main__':
    data_path = 'data/train'
    arch = 'RETFound_mae'
    chkpt_dir = hf_hub_download(repo_id="YukunZhou/RETFound_mae_meh", filename="RETFound_mae_meh.pth")

    print("==> Extracting features...")
    name_list, feature_list = get_feature_from_folders(data_path, chkpt_dir, arch=arch)

    print("==> Saving features to CSV...")
    df_feature = pd.DataFrame(feature_list)
    df_imgname = pd.DataFrame(name_list, columns=["name"])
    df_visualization = pd.concat([df_imgname, df_feature], axis=1)

    column_names = ["feature_{}".format(i) for i in range(df_feature.shape[1])]
    df_visualization.columns = ["name"] + column_names
    df_visualization.to_csv("Feature.csv", index=False)
    print("✅ Done. Features saved to Feature.csv")


import shap
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import external.RETFound_MAE.models_vit as models

# ---- 1. Model na CUDA ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.RETFound_mae(global_pool=True).to(device)
model.eval()

# ---- 2. Wczytaj i przetwórz obraz ----
img_path = "data/test/dseveredr/0edadb2aa127.png"
img = Image.open(img_path).convert("RGB").resize((224, 224))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

# ---- 3. Background: tylko 1–2 kopie (żeby nie pierdolnęło VRAMem) ----
background = img_tensor.repeat(2, 1, 1, 1)  # możesz też spróbować z 1

# ---- 4. SHAP GradientExplainer ----
explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(img_tensor)

# ---- 5. Przygotowanie do SHAP image_plot (na CPU!) ----
unnormalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

img_vis = unnormalize(img_tensor[0].detach().cpu()).clamp(0, 1).permute(1, 2, 0).numpy()

# ---- 6. SHAP wykres ----
shap.image_plot(shap_values, [img_vis])

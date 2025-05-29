import torch
from captum.attr import LayerGradCam
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import external.RETFound_MAE.models_vit as models
from shap_method import img_tensor

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.RETFound_mae(global_pool=True).to(device)
# Załaduj model i przygotuj obraz jak wcześniej
model.eval()

# Wybierz warstwę, na której chcesz Grad-CAM (zwykle ostatnia konwolucyjna lub transformer block)
layer = model.blocks[-1].norm2  # przykład, zależy od modelu

# Inicjuj Grad-CAM
gradcam = LayerGradCam(model, layer)

class_idx = 2
# Obraz - img_tensor [1,3,224,224], najlepiej na CPU, albo GPU jeśli masz
with torch.no_grad():
    attributions = gradcam.attribute(img_tensor, target=class_idx)  # podaj index klasy, którą chcesz wyjaśnić

# Upsample attributions do rozmiaru obrazu
attributions = torch.nn.functional.interpolate(attributions, size=(224,224), mode='bilinear', align_corners=False)

# Konwertuj do numpy i wyświetl
attr_np = attributions.squeeze().cpu().detach().numpy()

# Wizualizacja
plt.imshow(attr_np, cmap='jet', alpha=0.5)
plt.imshow(img_tensor.squeeze().permute(1, 2, 0).cpu(), alpha=0.5)
plt.axis('off')
plt.show()

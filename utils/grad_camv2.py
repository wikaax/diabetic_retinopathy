import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from captum.attr import LayerGradCam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(device)
model.eval()

img_path = "data/train/eproliferativedr_augmented/fb696a8e055a_aug160.png"
img = Image.open(img_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensor = transform(img).unsqueeze(0).to(device)

target_layer = model.layer4[-1].conv3
gradcam = LayerGradCam(model, target_layer)

output = model(input_tensor)
predicted_class = torch.argmax(output, dim=1).item()
print(f"Predykowana klasa: {predicted_class}")

# Grad-CAM attribution
attribution = gradcam.attribute(input_tensor, target=predicted_class)
attribution = attribution.squeeze().cpu().detach().numpy()
attribution = np.maximum(attribution, 0)
attribution = attribution / attribution.max()

heatmap = cv2.resize(attribution, (224, 224))

original_img = np.array(img.resize((224, 224))).astype(np.float32) / 255.0

heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
heatmap_color = heatmap_color[..., ::-1] / 255.0  # BGR to RGB

overlay = 0.5 * original_img + 0.5 * heatmap_color
overlay = np.clip(overlay, 0, 1)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(original_img)
ax[0].axis('off')
ax[0].set_title("Original Image")

ax[1].imshow(overlay)
ax[1].axis('off')
ax[1].set_title("Grad-CAM Overlay")

plt.tight_layout()
plt.show()

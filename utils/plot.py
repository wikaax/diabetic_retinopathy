import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

Z = np.load("logs_trans/retfound_features.npy")
y = np.load("labels.npy")

scaler = StandardScaler()
Z_scaled = scaler.fit_transform(Z)


pca = PCA(n_components=2)
Z_pca = pca.fit_transform(Z)

tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
Z_tsne = tsne.fit_transform(Z_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=y, cmap='tab10', s=20, alpha=0.7)
plt.title("PCA embedding visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.colorbar(scatter, label='Klasa')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
scatter = plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
plt.title("t-SNE embedding visualization")
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")
plt.colorbar(scatter, label="Class")
plt.grid(True)
plt.tight_layout()
plt.show()
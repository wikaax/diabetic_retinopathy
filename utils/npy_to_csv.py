import numpy as np
import pandas as pd

# Załaduj dane
Z = np.load("logs_trans/retfound_features.npy")  # (n_samples, n_features)
y = np.load("labels.npy")               # (n_samples,)

# Sklejamy embeddingi + etykiety
data = np.hstack([Z, y.reshape(-1, 1)])

# Tworzymy nazwy kolumn
n_features = Z.shape[1]
columns = [f'feat_{i}' for i in range(n_features)] + ['label']

# Tworzymy DataFrame
df = pd.DataFrame(data, columns=columns)

# Zapisujemy do CSV
df.to_csv("retfound_embeddings.csv", index=False)
print("Plik CSV zapisany jako resnet50_embeddings.csv")

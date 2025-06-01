import numpy as np
import pandas as pd

Z = np.load("medvit_embeddings.npy")
y = np.load("labels.npy")

data = np.hstack([Z, y.reshape(-1, 1)])

n_features = Z.shape[1]
columns = [f'feat_{i}' for i in range(n_features)] + ['label']

df = pd.DataFrame(data, columns=columns)

df.to_csv("medvit_embeddings.csv", index=False)
print("saved")

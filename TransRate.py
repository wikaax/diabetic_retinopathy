import pandas as pd
import numpy as np

df = pd.read_csv('Swin_Features.csv')

Z = df.drop(columns=['name']).values

y = df['name'].apply(lambda x: x.split('/')[0]).values

from sklearn.preprocessing import LabelEncoder, StandardScaler

le = LabelEncoder()
y_encoded = le.fit_transform(y)


def coding_rate(Z, eps=1e-2):
    n, d = Z.shape
    M = np.eye(d) + (1/(n*eps)) * Z.T @ Z
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        print("Warning: non-positive determinant!")
        return np.nan
    return 0.5 * logdet

def transrate(Z, y, eps):
    Z = Z - np.mean(Z, axis=0, keepdims=True)
    RZ = coding_rate(Z, eps)
    RZY = 0.0
    K = int(y.max()) + 1

    for i in range(K):
        class_samples = Z[(y == i).flatten()]
        RZY += coding_rate(class_samples, eps)

    return RZ - RZY / K

def h_score(Z, y, eps):
    K = int(y.max()) + 1
    h = 0.0
    for i in range(K):
        Zk = Z[(y == i).flatten()]
        n_k = Zk.shape[0]
        if n_k == 0:
            continue
        M = np.eye(Zk.shape[1]) + (1/(n_k * eps)) * Zk.T @ Zk
        sign, logdet = np.linalg.slogdet(M)
        if sign <= 0:
            print(f"Warning: non-positive determinant for class {i}")
            continue
        h += logdet
    return h / K


n, d = Z.shape
eps = 1e-4
M = np.eye(d) + (1/(n*eps)) * Z.T @ Z
sign, logdet = np.linalg.slogdet(M)
print("Logdet:", logdet)
scaler = StandardScaler()
Z_scaled = scaler.fit_transform(Z)
score = transrate(Z_scaled, y_encoded, eps)
print(f"TransRate score: {score}")

score_h = h_score(Z, y_encoded, eps=1e-1)
print(f"H-score: {score_h}")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def coding_rate(Z, eps=1e-4):
    n, d = Z.shape
    M = np.eye(d) + (1 / (n * eps)) * Z.T @ Z
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        print("Warning: non-positive determinant!")
        return np.nan
    return 0.5 * logdet

def transrate(Z, y, eps=1e-4):
    Z = Z - np.mean(Z, axis=0, keepdims=True)
    RZ = coding_rate(Z, eps)
    RZY = 0.0
    K = int(y.max()) + 1

    for i in range(K):
        class_samples = Z[y == i]
        RZY += coding_rate(class_samples, eps)

    return RZ - RZY / K


def h_score(Z, y):
    """
    Z: numpy array (n_samples, n_features) - cechy
    y: numpy array (n_samples,) - etykiety klasowe (liczby całkowite)

    Zwraca: float - wartość H-score
    """
    n, d = Z.shape
    classes = np.unique(y)

    mean_all = np.mean(Z, axis=0)

    Sb = np.zeros((d, d))
    Sw = np.zeros((d, d))

    for c in classes:
        Zc = Z[y == c]
        nc = Zc.shape[0]
        mean_c = np.mean(Zc, axis=0)

        mean_diff = (mean_c - mean_all).reshape(-1, 1)
        Sb += nc * (mean_diff @ mean_diff.T)

        diff = Zc - mean_c
        Sw += diff.T @ diff

    epsilon = 1e-10
    score = np.trace(Sb) / (np.trace(Sw) + epsilon)

    return score


def logdet_divergence(A, B, epsilon=1e-4):
    d = A.shape[0]

    A = make_positive_definite(A, epsilon)
    B = make_positive_definite(B, epsilon)

    trace_term = np.trace(A @ np.linalg.inv(B))

    sign_A, logdet_A = np.linalg.slogdet(A)
    sign_B, logdet_B = np.linalg.slogdet(B)

    if sign_A <= 0 or sign_B <= 0:
        print("Warning: non-positive determinant in LogDet divergence!")
        return np.nan

    logdet_term = logdet_A - logdet_B
    divergence = trace_term - logdet_term - d
    return divergence

def compute_logdet_divergence_score(Z, y):
    n, d = Z.shape
    classes = np.unique(y)

    cov_all = np.cov(Z, rowvar=False)
    cov_all = make_positive_definite(cov_all)

    cov_classes = np.zeros((d, d))
    for c in classes:
        Zc = Z[y == c]
        cov_c = np.cov(Zc, rowvar=False)
        cov_c = make_positive_definite(cov_c)
        cov_classes += cov_c

    cov_classes /= len(classes)
    cov_classes = make_positive_definite(cov_classes)

    return logdet_divergence(cov_all, cov_classes)

def make_positive_definite(matrix, eps=1e-4):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals[eigvals < eps] = eps
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

import numpy as np

Z = np.load('medvit_embeddings.npy')
y = np.load('labels.npy')
print("y shape: ", y.shape)
print(y)
scaler = MinMaxScaler()
Z_norm = scaler.fit_transform(Z)

eps = 1e-3
score = transrate(Z, y, eps)
print(f"TransRate score: {score}")

h = h_score(Z, y)
print(f"H-score: {h}")

logdet_div = compute_logdet_divergence_score(Z, y)
print(f"LogDet Divergence: {logdet_div}")
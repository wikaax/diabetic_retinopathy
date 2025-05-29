import os
import numpy as np

categories = ['anodr', 'bmilddr_augmented', 'cmoderatedr_augmented', 'dseveredr_augmented', 'eproliferativedr_augmented']
base_path = 'data/train'

labels = []

for idx, category in enumerate(categories):
    folder_path = os.path.join(base_path, category)
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} nie istnieje.")
        continue

    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    labels += [idx] * len(files)

labels = np.array(labels)
print("Labels shape:", labels.shape)
print("Przykładowe etykiety:", labels[:10])

np.save("labels.npy", labels)

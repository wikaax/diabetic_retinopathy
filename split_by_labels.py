import os
import shutil
import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "data/raw_split"
CSV_PATH = "messidor_data.csv"

for i in range(5):
    os.makedirs(os.path.join(OUT_DIR, str(i)), exist_ok=True)

df = pd.read_csv(CSV_PATH)

df = df[df['diagnosis'].isin([0, 1, 2, 3, 4])]

for idx, row in df.iterrows():
    filename = row['id_code']
    label = str(row['diagnosis'])

    src_path = os.path.join(RAW_DIR, filename)
    dst_path = os.path.join(OUT_DIR, label, filename)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"⚠️ Nie znaleziono pliku: {filename}")

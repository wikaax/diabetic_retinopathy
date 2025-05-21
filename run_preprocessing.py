import os
from preprocessing import preprocess_image
from tqdm import tqdm

IN_DIR = "data/raw_split"
OUT_DIR = "data/processed"

os.makedirs(OUT_DIR, exist_ok=True)

for label in os.listdir(IN_DIR):
    input_class_dir = os.path.join(IN_DIR, label)
    output_class_dir = os.path.join(OUT_DIR, label)
    os.makedirs(output_class_dir, exist_ok=True)

    for img_file in tqdm(os.listdir(input_class_dir), desc=f"Przetwarzanie klasy {label}"):
        input_path = os.path.join(input_class_dir, img_file)
        output_path = os.path.join(output_class_dir, img_file)

        try:
            img = preprocess_image(input_path)
            img.save(output_path)
        except Exception as e:
            print(f"❌ Błąd w {img_file}: {e}")

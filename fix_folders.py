import os
import shutil

base_path = r"E:\detekcja_retinopatii_cukrzycowej\data\processed"
old_class = "4"
new_class = "3"

src_dir = os.path.join(base_path, old_class)
dst_dir = os.path.join(base_path, new_class)

os.makedirs(dst_dir, exist_ok=True)

for filename in os.listdir(src_dir):
    src_file = os.path.join(src_dir, filename)
    dst_file = os.path.join(dst_dir, filename)

    if os.path.isfile(src_file):
        shutil.move(src_file, dst_file)

os.rmdir(src_dir)

print("✅ Przeniesiono zdjęcia z klasy 4 → 3 i usunięto folder.")

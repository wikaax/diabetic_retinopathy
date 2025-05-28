import os
import random
import shutil
from PIL import Image, ImageEnhance

def count_files_in_subfolders(base_path):
    categories = ['anodr', 'bmilddr', 'cmoderatedr', 'dseveredr', 'eproliferativedr']
    counts = {}

    for category in categories:
        folder_path = os.path.join(base_path, category)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            counts[category] = len(files)
        else:
            counts[category] = 0
            print(f"Uwaga: folder {folder_path} nie istnieje!")

    return counts

def augment_image(img):
    # Losowo wybieramy augmentację, możesz dodawać lub usuwać
    choice = random.choice(['rotate', 'flip_horizontal', 'flip_vertical', 'brightness', 'crop'])

    if choice == 'rotate':
        angle = random.uniform(-15, 15)
        return img.rotate(angle, expand=True)

    elif choice == 'flip_horizontal':
        return img.transpose(Image.FLIP_LEFT_RIGHT)

    elif choice == 'flip_vertical':
        return img.transpose(Image.FLIP_TOP_BOTTOM)

    elif choice == 'brightness':
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.7, 1.3)  # jasność od 70% do 130%
        return enhancer.enhance(factor)

    elif choice == 'crop':
        width, height = img.size
        crop_percent = random.uniform(0.85, 0.95)  # przycinamy 5-15% obrazu losowo
        new_w = int(width * crop_percent)
        new_h = int(height * crop_percent)
        left = random.randint(0, width - new_w)
        top = random.randint(0, height - new_h)
        return img.crop((left, top, left + new_w, top + new_h)).resize((width, height))

    return img  # fallback

def oversample_with_augmentation(base_path, target_count=1010):
    categories = ['anodr', 'bmilddr', 'cmoderatedr', 'dseveredr', 'eproliferativedr']

    for category in categories:
        folder_path = os.path.join(base_path, category)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} nie istnieje, pomijam.")
            continue

        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        current_count = len(files)

        if current_count == 0:
            print(f"Brak plików w {folder_path}, pomijam oversampling.")
            continue

        if current_count >= target_count:
            print(f"{category}: już jest {current_count} plików, oversampling nie jest potrzebny.")
            continue

        oversampled_folder = folder_path + "_augmented"
        os.makedirs(oversampled_folder, exist_ok=True)

        # Kopiujemy oryginały
        for f in files:
            shutil.copy2(os.path.join(folder_path, f), os.path.join(oversampled_folder, f))

        needed = target_count - current_count

        for i in range(needed):
            file_to_copy = random.choice(files)
            img_path = os.path.join(folder_path, file_to_copy)

            try:
                img = Image.open(img_path)
                img_aug = augment_image(img)

                name, ext = os.path.splitext(file_to_copy)
                new_name = f"{name}_aug{i}{ext}"

                save_path = os.path.join(oversampled_folder, new_name)
                img_aug.save(save_path)
            except Exception as e:
                print(f"Nie udało się załadować/zmodyfikować {img_path}: {e}")

        print(f"{category}: augmentacja zrobiona, teraz około {target_count} plików w {oversampled_folder}")

if __name__ == "__main__":
    base_folder = "data/train"
    counts = count_files_in_subfolders(base_folder)
    for category, count in counts.items():
        print(f"{category}: {count} plików")

    print("\nOversampling in progress")
    oversample_with_augmentation(base_folder, target_count=1010)

import os
import shutil
import yaml
from sklearn.model_selection import train_test_split

# Chemins
IMAGE_DIR = "C:/Users/yvan_/OneDrive/Documents/Trich/datasets/images"
LABEL_DIR = "C:/Users/yvan_/OneDrive/Documents/Trich/datasets/labels"
OUTPUT_DIR = "C:/Users/yvan_/OneDrive/Documents/Trich/datasets/yolo_dataset"

# Créer la structure
os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/images/val", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/images/test", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/val", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels/test", exist_ok=True)

# Fonction pour vérifier l'existence du label
def has_label(image_file):
    base_name = os.path.splitext(image_file)[0]
    label_file = f"{base_name}.txt"
    return os.path.exists(os.path.join(LABEL_DIR, label_file))

# Lister uniquement les images avec labels
valid_image_files = [f for f in os.listdir(IMAGE_DIR) 
                    if f.endswith(('.jpg', '.jpeg', '.png')) 
                    and has_label(f)]

print(f"Total images with labels: {len(valid_image_files)}")

# Stratified split (80% train, 10% val, 10% test)
train, test = train_test_split(valid_image_files, test_size=0.2, random_state=42)
val, test = train_test_split(test, test_size=0.5, random_state=42)

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

# Copier les fichiers (uniquement ceux avec labels)
def copy_files(files, split):
    for img_file in files:
        base_name = os.path.splitext(img_file)[0]
        
        # Copier image
        shutil.copy(
            os.path.join(IMAGE_DIR, img_file),
            os.path.join(OUTPUT_DIR, "images", split, img_file)
        )
        
        # Copier label
        label_file = f"{base_name}.txt"
        shutil.copy(
            os.path.join(LABEL_DIR, label_file),
            os.path.join(OUTPUT_DIR, "labels", split, label_file)
        )

copy_files(train, "train")
copy_files(val, "val")
copy_files(test, "test")

# Créer dataset.yaml
data_yaml = {
    "path": os.path.abspath(OUTPUT_DIR),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {0: "pulling", 1: "not_pulling"},
    "nc": 2
}

with open(f"{OUTPUT_DIR}/dataset.yaml", "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

# Vérifier les images sans labels
all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
images_without_labels = [f for f in all_images if not has_label(f)]

if images_without_labels:
    print(f"\n⚠️ Attention: {len(images_without_labels)} images sans labels trouvées:")
    for img in images_without_labels[:5]:  # Afficher seulement les 5 premières
        print(f" - {img}")
    if len(images_without_labels) > 5:
        print(f" - ... (total {len(images_without_labels)} images)")
else:
    print("\n✅ Toutes les images ont des labels associés!")

print("\nDataset préparé avec succès!")
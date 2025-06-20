import os
import torch
from ultralytics import YOLO

# Configuration
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # Optimisation pour CPU

MODEL_SIZE = "s"  # Version small pour meilleure précision
EPOCHS = 200      # Plus d'epochs pour le grand dataset
BATCH_SIZE = 32   # Batch size augmenté
IMG_SIZE = 640
DATA_PATH = "datasets/yolo_dataset/dataset.yaml"

# Chemin de sauvegarde
PROJECT_PATH = r"C:\Users\yvan_\OneDrive\Documents\Trich\weights"
MODEL_NAME = "trich_model_v2"

# Vérification GPU
print("="*50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
else:
    print("⚠️ ATTENTION: Aucun GPU détecté - L'entraînement sera lent!")
print("="*50)

# Augmentations spécifiques pour la détection de "pulling"
AUGMENTATIONS = {
    "degrees": 10.0,       # Rotation limitée
    "translate": 0.1,      # Translation modérée
    "scale": 0.4,          # Changement d'échelle
    "shear": 0.1,          # Cisaillement léger
    "perspective": 0.0004, # Perspective subtile
    "fliplr": 0.5,         # Retour horizontal
    "mosaic": 0.8,         # Mosaic augmentation
    "mixup": 0.1,          # Mixup léger
    "copy_paste": 0.2,     # Important pour "pulling"
    "hsv_h": 0.01,         # Teinte
    "hsv_s": 0.6,          # Saturation
    "hsv_v": 0.4,          # Valeur
    "erasing": 0.3,        # Effacement aléatoire
}

def main():
    # Charger le modèle pré-entraîné
    model = YOLO(f"yolov8{MODEL_SIZE}.pt")
    
    # Configurer le device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🚀 Démarrage de l'entraînement sur {'GPU' if device != 'cpu' else 'CPU'}")
    
    # Créer le dossier de sauvegarde
    os.makedirs(PROJECT_PATH, exist_ok=True)
    
    # Paramètres d'entraînement optimisés
    train_args = {
        "data": DATA_PATH,
        "epochs": EPOCHS,
        "batch": BATCH_SIZE,
        "imgsz": IMG_SIZE,
        "device": device,
        "patience": 40,      # Plus de patience pour grand dataset
        "optimizer": "AdamW",
        "lr0": 0.0001,       # Taux d'apprentissage plus bas
        "lrf": 0.01,         # Facteur final de learning rate
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "box": 7.5,          # Poids plus élevé pour la perte de localisation
        "cls": 0.5,          # Poids normal pour la classification
        "dfl": 1.5,          # Distribution Focal Loss
        "label_smoothing": 0.1,
        "workers": 8 if device != "cpu" else 4,
        "project": PROJECT_PATH,
        "name": MODEL_NAME,
        "exist_ok": True,
        "verbose": True,
        "close_mosaic": 10,  # Désactive mosaic pour les dernières epochs
    }
    
    # Ajouter les augmentations
    train_args.update(AUGMENTATIONS)
    
    # Ajouter les optimisations GPU si disponible
    if device != "cpu":
        train_args.update({
            "half": True,    # Précision mixte FP16
            "amp": True,     # Automatic Mixed Precision
        })
    
    # Démarrer l'entraînement
    results = model.train(**train_args)
    
    # Évaluation du modèle
    best_model_path = os.path.join(PROJECT_PATH, MODEL_NAME, "weights", "best.pt")
    best_model = YOLO(best_model_path)
    metrics = best_model.val()
    print(f"✅ Entraînement terminé! mAP50: {metrics.box.map50:.4f}")
    
    # Rapport de classes
    if hasattr(metrics, 'results_dict'):
        print("\nRapport de performance par classe:")
        print(f"- Pulling (0): AP50 = {metrics.results_dict.get('metrics/ap50_0', 0):.4f}")
        print(f"- Not Pulling (1): AP50 = {metrics.results_dict.get('metrics/ap50_1', 0):.4f}")
    
    # Export pour déploiement
    export_path = os.path.join(PROJECT_PATH, MODEL_NAME, "weights", "best.onnx")
    best_model.export(format="onnx", imgsz=IMG_SIZE, simplify=True)
    print(f"Modèle exporté: {export_path}")

if __name__ == "__main__":
    main()
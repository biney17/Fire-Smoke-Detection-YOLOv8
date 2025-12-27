from ultralytics import YOLO
import cv2
from tkinter import Tk, filedialog

def main():
    # 1️⃣ Charger le modèle entraîné
    model = YOLO("runs/detect/train/weights/best.pt")

    # 2️⃣ Ouvrir une fenêtre pour choisir une image
    Tk().withdraw()  # cacher la fenêtre principale
    image_path = filedialog.askopenfilename(
        title="Choisir une image",
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )

    if not image_path:
        print("❌ Aucune image sélectionnée")
        return

    # 3️⃣ Faire la prédiction
    results = model.predict(
        source=image_path,
        conf=0.25,
        save=True,
        show=False
    )

    # 4️⃣ Afficher l'image avec les prédictions
    img = results[0].plot()
    cv2.imshow("Prediction YOLOv8 - Feu & Fumée", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

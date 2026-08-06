import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

@st.cache_resource
def load_model():
    return YOLO("runs/detect/train/weights/best.pt")

model = load_model()

st.title("🔥 Détection Feu & Fumée - YOLOv8")

tab1, tab2 = st.tabs(["📷 Image", "🎥 Vidéo"])

# ---------- ONGLET IMAGE ----------
with tab1:
    uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"], key="img")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        conf = st.slider("Seuil de confiance", 0.0, 1.0, 0.25, 0.05, key="conf_img")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Originale", use_container_width=True)
        
        with st.spinner("Détection..."):
            results = model.predict(source=image, conf=conf, save=False)
        
        annotated = results[0].plot()[:, :, ::-1]
        with col2:
            st.image(annotated, caption="Résultat", use_container_width=True)

# ---------- ONGLET VIDÉO ----------
with tab2:
    uploaded_video = st.file_uploader("Choisir une vidéo", type=["mp4", "avi", "mov"], key="vid")
    conf_vid = st.slider("Seuil de confiance", 0.0, 1.0, 0.25, 0.05, key="conf_vid")
    
    if uploaded_video is not None:
        # Sauvegarde temporaire, car OpenCV a besoin d'un chemin fichier
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        st.video(uploaded_video)  # aperçu de la vidéo originale
        
        start_button = st.button("Lancer la détection")
        
        if start_button:
            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()  # placeholder qui se rafraîchit à chaque frame
            progress_bar = st.progress(0)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            
            stop_button = st.button("⏹️ Arrêter")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # frame est en BGR (OpenCV) -> compatible direct avec predict
                results = model.predict(source=frame, conf=conf_vid, save=False, verbose=False)
                annotated_frame = results[0].plot()  # reste en BGR
                annotated_frame_rgb = annotated_frame[:, :, ::-1]  # BGR -> RGB pour st.image
                
                stframe.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                
                frame_count += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
            
            cap.release()
            os.unlink(video_path)  # nettoyage du fichier temporaire
            st.success("✅ Traitement terminé")
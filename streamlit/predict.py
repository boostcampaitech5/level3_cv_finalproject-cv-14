import streamlit as st
import cv2
from deepface import DeepFace

@st.cache_resource
def detect(img):
    try:
        aligned_img = DeepFace.extract_faces(img, detector_backend="mtcnn", enforce_detection=True)
        aligned_img = aligned_img[0]['face']
        rgb_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
        return rgb_img

    except ValueError as e:
        return st.error("얼굴 감지 중 오류가 발생했습니다. 얼굴을 인식할 수 없습니다.")
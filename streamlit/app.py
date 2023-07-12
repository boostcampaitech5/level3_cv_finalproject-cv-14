import streamlit as st
from deepface import DeepFace
from predict import detect
from utils import load_image
from streamlit_option_menu import option_menu
import io
import numpy as np
import glob
import sys

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

st.title("Face Recognition")
    
def main():
    with st.sidebar:
        selected = option_menu(None, ["Upload", "Face Detect", "Face Verify", "Face Find"], 
        icons=["bi bi-image", "bi bi-person-bounding-box", "bi bi-people-fill", "bi bi-search"], 
        menu_icon="cast", default_index=0)
        
    if selected == "Upload":
        uploaded_file = st.file_uploader("Choose an image",type=["jpg","jpeg","png"])
        
        if uploaded_file:
            # 업로드된 파일을 바이너리 데이터로 읽어들임
            img = load_image(uploaded_file.read())
            
            # 이미지를 session_state에 저장
            st.session_state["uploaded_image"] = img
            
        if "uploaded_image" in st.session_state:
            img = st.session_state["uploaded_image"]
            st.image(img)
        
    elif selected == "Face Detect":
        if "uploaded_image" in st.session_state:
            img = st.session_state["uploaded_image"]
            detect_img = detect(img)
            st.image(detect_img)
        else:
            st.warning("Please upload an image first.")
            
    elif selected == "Face Verify":
        pass
    elif selected == "Face Find":
        pass
        

if __name__ == '__main__':
    main()
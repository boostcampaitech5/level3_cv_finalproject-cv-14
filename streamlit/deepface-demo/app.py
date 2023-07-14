import streamlit as st
from deepface import DeepFace
from predict import detect, verify_face, find_face
from clustering import perform_kmeans_clustering, perform_dbscan_clustering, perform_hierarchical_clustering
from calculate import *
from utils import load_image, convert_to_static_url
from streamlit_option_menu import option_menu
import numpy as np
import glob
import sys
import pandas as pd
import pickle
from os import path


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

st.title("Face Recognition")
st.divider()

# cache 값 지우기
# if st.button("Clear All"):
#     # Clears all st.cache_resource caches:
#     st.cache_resource.clear()
#     st.cache_data.clear()


def menu_upload():
    uploaded_file = st.file_uploader("Choose an image",type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # 업로드된 파일을 바이너리 데이터로 읽어들임
        img = load_image(uploaded_file.read())
            
        # 이미지를 session_state에 저장
        st.session_state["uploaded_image"] = img
            
    if "uploaded_image" in st.session_state:
        img = st.session_state["uploaded_image"]
        st.image(img)
    else:
        st.warning("Please upload an image first.")


def menu_face_detect():
    if "uploaded_image" in st.session_state:
        img = st.session_state["uploaded_image"]
            
        select_backend = st.radio(
            "Select a detector backend",
            ["opencv", "ssd", "mtcnn", "retinaface"],
            horizontal=True,
            index=2)
        detect_img = detect(img, select_backend)
        st.image(detect_img)
    else:
        st.warning("Please upload an image first.")


def menu_face_verify():
    if "uploaded_image" in st.session_state:
        img = st.session_state["uploaded_image"]
            
        select_model = st.radio(
            "Select a model",
            ["VGG-Face", "Facenet", "Facenet512", "ArcFace"],
            horizontal=True,
            index=1)
            
        select_backend = st.radio(
            "Select a detector backend",
            ["mtcnn", "retinaface"],
            horizontal=True)
            
        select_metric = st.radio(
            "Select a similarity metric",
            ["cosine", "euclidean", "euclidean_l2"],
            horizontal=True,
            index=2)
            
        results = verify_face(img, select_model, select_backend, select_metric)

        df = pd.DataFrame(results)
        df["img_path"] = df["img_path"].apply(convert_to_static_url)
            
        st.dataframe(
            df,
            column_config={
            "verified": st.column_config.TextColumn(),
            "label": st.column_config.TextColumn(),
            "model": None,
            "detector_backend": None,
            "similarity_metric": None,
            "img_path": st.column_config.ImageColumn()
            },
            use_container_width=True,
            hide_index=True,
            column_order=("img_path", "label", "verified", "distance", "threshold", "time", "facial_areas")
            )
    else:
        st.warning("Please upload an image first.")


def menu_face_find():
    if "uploaded_image" in st.session_state:
        img = st.session_state["uploaded_image"]
        
        select_model = st.radio(
            "Select a model",
            ["VGG-Face", "Facenet", "Facenet512", "ArcFace"],
            horizontal=True,
            index=3)
            
        select_backend = st.radio(
            "Select a detector backend",
            ["mtcnn", "retinaface"],
            horizontal=True)
            
        select_metric = st.radio(
            "Select a similarity metric",
            ["cosine", "euclidean", "euclidean_l2"],
            horizontal=True,
            index=2)
            
        results = find_face(img, select_model, select_backend, select_metric)
            
        df = pd.DataFrame(results)
        df["identity"] = df["identity"].apply(convert_to_static_url)
        df["path"]=df["identity"]
        metric_str = f'{select_model}_{select_metric}'
            
        st.dataframe(
            df,
            column_config={
                "path": st.column_config.TextColumn(),
                "metric": st.column_config.TextColumn(metric_str),
                "source_x": None,
                "source_y": None,
                "source_w": None,
                "source_h": None,
                "identity": st.column_config.ImageColumn()
                },
                use_container_width=True,
                hide_index=True,
                column_order=("identity", "path", metric_str)
                )
        db_path = "/".join(results["identity"][0].split("/")[:-1])
        file_name = f"representations_{select_model}.pkl"
        file_name = file_name.replace("-", "_").lower()
        file_path = db_path + "/" + file_name
            
        # 파일이 존재하는 경우 선택 박스 생성
        if path.exists(file_path):
            # clustering 알고리즘 선택 박스
            clustering_algorithm = st.selectbox("Select a clustering algorithm", ["K-means", "DBSCAN", "Hierarchical Clustering"])
            # 선택 박스에 따른 동작 수행
            if clustering_algorithm == "K-means":
                # K-means 클러스터링 수행
                perform_kmeans_clustering(file_path)
            elif clustering_algorithm == "DBSCAN":
                 # DBSCAN 클러스터링 수행
                perform_dbscan_clustering(file_path)
            elif clustering_algorithm == "Hierarchical Clustering":
                # 계층적 클러스터링 수행
                perform_hierarchical_clustering(file_path)
                
            # similarity 알고리즘 선택 박스
            similarity_algorithm = st.selectbox("Calculate similarity", ["cosine", "euclidean", "euclidean_l2", "manhattan"])
            # 선택 박스에 따른 동작 수행
            calculate_similarity(file_path, img, select_model, select_backend, similarity_algorithm)
            
        else:
            st.warning("The file does not exist.")        
    else:
        st.warning("Please upload an image first.")
    
    
    
def main():
    with st.sidebar:
        selected = option_menu(None, ["Upload", "Face Detect", "Face Verify", "Face Find"], 
        icons=["bi bi-image", "bi bi-person-bounding-box", "bi bi-people-fill", "bi bi-search"], 
        menu_icon="cast", default_index = 0)
        
    if selected == "Upload":
        menu_upload()
        
    elif selected == "Face Detect":
        menu_face_detect()
            
    elif selected == "Face Verify":
        menu_face_verify()
            
    elif selected == "Face Find":
        menu_face_find()
        

if __name__ == '__main__':
    main()
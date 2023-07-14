import streamlit as st
import cv2
from deepface import DeepFace
from utils import load_path


@st.cache_resource
def detect(img, select_backend):
    try:
        aligned_img = DeepFace.extract_faces(img, 
                                             detector_backend=select_backend, 
                                             enforce_detection=True)
        
        aligned_img = aligned_img[0]['face']
        rgb_img = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
        return rgb_img

    except ValueError as e:
        return st.error("얼굴 감지 중 오류가 발생했습니다. 얼굴을 인식할 수 없습니다.")


@st.cache_resource  
def verify_face(img, select_model, select_backend, select_metric):
    try:
        train_img_path = "./static/celeb_40/cut_train/"
        img_paths = load_path(train_img_path)
        verify_results = []
       
        progress_bar = st.progress(0)
        progress_text = st.empty()
        for i, img_path in enumerate(img_paths):
            result = DeepFace.verify(img, 
                                     img_path, 
                                     model_name=select_model, 
                                     detector_backend=select_backend, 
                                     distance_metric=select_metric, 
                                     enforce_detection=True)
           
            if result["verified"] == True:
                img_name = img_path.split("/")[-1] 
                img_name_without_extension = img_name.split(".")[0]
                
                result["img_path"] = img_path
                result["label"] = img_name_without_extension
                
                verify_results.append(result)
                
            # Update progress bar and text
            progress = (i + 1) / len(img_paths)
            progress_bar.progress(progress)
            progress_text.text(f"Progress: {int(progress * 100)}%")
            
        return verify_results

    except ValueError as e:
        return st.error("얼굴 감지 중 오류가 발생했습니다. 얼굴을 인식할 수 없습니다.")

@st.cache_resource  
def find_face(img, select_model, select_backend, select_metric):
    try:
        find_results = DeepFace.find(img_path=img,
                                db_path="./static/celeb_40/cut_train/",
                                model_name=select_model,
                                detector_backend=select_backend,
                                distance_metric=select_metric,
                                enforce_detection=True)[0]
        
        return find_results

    except ValueError as e:
        return st.error("얼굴 감지 중 오류가 발생했습니다. 얼굴을 인식할 수 없습니다.")


@st.cache_resource 
def represent_face(img, select_model, select_backend):
    try:
        represent_results = DeepFace.represent(img_path=img,
                       model_name=select_model,
                       enforce_detection=True,
                       detector_backend="skip"
                       )
        
        return represent_results

    except ValueError as e:
        return st.error("얼굴 감지 중 오류가 발생했습니다. 얼굴을 인식할 수 없습니다.")
    
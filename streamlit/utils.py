import cv2
import numpy as np

def load_image(image_bytes):
    # 바이너리 데이터로 이미지 열기
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 이미지 크기를 비율에 맞게 조정
    max_dim = max(img.shape[0], img.shape[1])  # 이미지의 가장 긴 길이를 기준으로 비율 조정
    resize_ratio = 400 / max_dim  # 400은 원하는 기준 길이로 수정 가능
    new_width = int(img.shape[1] * resize_ratio)
    new_height = int(img.shape[0] * resize_ratio)
    resized_img = cv2.resize(img, (new_width, new_height))

    # # 이미지 색상 변환 (BGR to RGB)
    rgb_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    
    return rgb_image

   
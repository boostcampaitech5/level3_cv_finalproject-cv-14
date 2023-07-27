from flask import Flask, request, jsonify
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
import numpy as np
from torchvision import transforms
import requests
from pymongo import MongoClient
from flask_cors import CORS
import logging
import sys 
from datetime import datetime

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

# 모델 초기화
detection_model = MTCNN(image_size=160, margin=0.6, min_face_size=20, thresholds=[0.6,0.7,0.7], factor= 0.709, keep_all=True) 
single_detection_model = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709)
facenet_model = InceptionResnetV1(classify=False, pretrained='vggface2')
threshold = 0.3046
facenet_model.eval()


def save_to_mongodb(url, embedding, user_email):
    try:
        # MongoDB 연결 설정
        client = MongoClient("mongodb+srv://admin:qwer1234@cluster0.8svfarz.mongodb.net/memory-studio?retryWrites=true&w=majority")  # 여기에 MongoDB 연결 정보를 입력해주세요.
        # db = client["memory-studio"]  # 여기에 실제 사용할 데이터베이스 이름을 입력해주세요.
        collection = client['memory-studio']['users']  # 여기에 실제 사용할 컬렉션 이름을 입력해주세요.

        # 현재 시간을 가져와서 uploadTime 필드에 저장
        current_time = datetime.now()
        # 이미지 URL과 임베딩 벡터를 MongoDB에 저장
        document  = {
            'imagePath': url,
            'embeddingVector': embedding,
            'uploadTime': current_time
        }
        user = collection.find_one({'email': user_email})
        if user:
            # 이미 사용자가 존재하는 경우, 이미지 정보를 추가합니다.
            collection.update_one({'email': user_email}, {'$push': {'imagePaths': document}})
        else:
            # 사용자가 존재하지 않는 경우, 에러를 반환합니다.
            raise Exception("사용자가 존재하지 않습니다.")

        client.close()
    except Exception as e:
        print(f"MongoDB에 데이터 저장 중 오류 발생: {str(e)}")
        # 사용자가 존재하지 않는 경우 에러를 반환합니다.
        return str(e)
    # 성공적으로 저장되었을 경우 None을 반환합니다.
    return None

def get_image_from_s3(url):
    try:
        # URL에서 이미지 데이터를 가져오기
        response = requests.get(url)
        response.raise_for_status()

        # 가져온 데이터를 NumPy 배열로 변환
        image_np = np.asarray(bytearray(response.content), dtype=np.uint8)

        # NumPy 배열을 OpenCV 이미지로 변환
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        return img

    except Exception as e:
        print(f"Error loading image from URL: {str(e)}")
        return None

def get_embedding(model, image_path=None, feature=None) :
    # image_path로 기입된 경우와 image자체로 기입된 경우를 나눈다.
    if image_path is not None :
        img = get_image_from_s3(image_path)
    else : 
        img = feature
        
    # 이미지의 형태(shape)를 확인하기 위해 디버그 출력을 추가합니다.
    print(f"이미지 형태: {img.shape}")

    # dataset을 거치지 않고 나온 이미지는 3차원이기 때문에 모델에 넣어줄 수 있게 4차원 변환
    if len(img.shape) == 3:         
        img = img.unsqueeze(0)
        
    with torch.no_grad():  # Use torch.no_grad() to disable gradient computation
        embedding = model(img)
        embedding = embedding.detach().cpu().numpy()    # embedding 연산은 CPU로 진행하기 위해 변환

    return embedding

@app.route("/process_image", methods=["POST"])
def process_image():
    try:
        data = request.json
        print(f"data: {data}")
        file_urls = data.get("fileUrls", [])
        user_email = data.get("userEmail")

        embeddings_dict = {}  # 이미지 경로와 임베딩 벡터를 저장할 딕셔너리
        
        # 이미지들에 대한 얼굴 탐지 및 임베딩 벡터 계산
        for url in file_urls:
            img = get_image_from_s3(url)
            if img is None:
                print(f"Cannot load image from S3: {url}")
                continue

            with torch.no_grad():
                feature = detection_model(img)
                if feature is None:
                    print(f"Cannot detect face in the image: {url}")
                    continue

            # 임베딩 벡터 계산 및 저장
            embedding = get_embedding(facenet_model, feature=feature)
            embeddings_dict[url] = embedding.tolist()
            
            # 임베딩 벡터와 이미지 URL을 MongoDB에 저장
            save_to_mongodb(url, embedding.tolist(), user_email)

        return jsonify(embeddings_dict)

    except Exception as e:
        app.logger.error(f"Error processing images: {str(e)}")
        return jsonify({'error': str(e)}), 500


#//////////////////////////////////////////


def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist




# 거리를 비교해서, 적절한 image를 set형태로 return
def get_result (key, enrolled_DB, threshold, album : dict) :
    key_image_set = set()
    key_image_embedding = enrolled_DB[key]
    for image in album.keys() :
        embeddings = album[image] # (n, 512)
        dis = distance(key_image_embedding, embeddings, 1)
        result = dis <= threshold  # result [True, False, False] 
        if any(result) :          # any => 하나라도 True가 있으면
            key_image_set.add(image)

    return key_image_set


# 이미지 유사도 계산 엔드포인트
@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        print(f"data: {data}")
        file_url = data.get("fileUrl")
        user_email = data.get("userEmail")
        userImagePaths = data.get("userImagePaths")
        
        key_dict = {}  # 이미지 경로와 임베딩 벡터를 저장할 딕셔너리 
        
        # 이미지들에 대한 얼굴 탐지 및 임베딩 벡터 계산
        
        img = get_image_from_s3(file_url)
        if img is None:
            print(f"Cannot load image from S3: {url}")
            

        with torch.no_grad():
            feature = single_detection_model(img)
            if feature is None:
                print(f"Cannot detect face in the image: {url}")
                

        # 임베딩 벡터 계산 및 저장
        embedding = get_embedding(facenet_model, feature=feature)
        key_dict[url] = embedding.tolist()
        print(key_dict)

        final_classification = get_result(file_url, key_dict, threshold, userImagePaths)   
        # return jsonify(embeddings_dict)

    except Exception as e:
        app.logger.error(f"Error processing images: {str(e)}")
        return jsonify({'error': str(e)}), 500

    



@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/predict', methods=['POST'])
def predict():
    # TODO : load image
    # TODO : image -> tensor
    # TODO : prediction
    # TODO : return json
    return jsonify({'result':1})


    

if __name__ == '__main__':
    app.run()

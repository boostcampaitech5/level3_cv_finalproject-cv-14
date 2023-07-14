import pickle
import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from predict import represent_face
import pandas as pd
from utils import load_path
import os
from collections import defaultdict
import math


def calculate(eu_top_rank_list, train_label):
    count = 0
    for dis, label in eu_top_rank_list:
        if label == train_label:
            count += 1
    return count


def distance(embedding1, embedding2, similarity_algorithm):
    ["cosine", "euclidean", "euclidean_l2", "manhattan"]
    if similarity_algorithm == "cosine":
        dot = np.sum(np.multiply(embedding1, embedding2))
        norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
        return dist
    elif similarity_algorithm == "euclidean":
        diff = np.subtract(embedding1, embedding2)
        dist = np.sum(np.square(diff),1)
        return dist
    elif similarity_algorithm == "euclidean_l2":
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)
        return np.linalg.norm(embedding1 - embedding2)
    else:
        return cityblock(embedding1, embedding2)


def calculate_similarity(file_path, img, select_model, select_backend, similarity_algorithm):
    test_img_path = "./static/celeb_40/cut_test/"
    test_img_paths = load_path(test_img_path)
    
    train_img_path = "./static/celeb_40/cut_train/"
    train_img_paths = load_path(train_img_path)
    
    test_embeddings = []
    test_labels = []
    for img_path in test_img_paths:
        represent_results = represent_face(img_path, select_model, select_backend)
        test_embeddings.append(represent_results[0]["embedding"])
        test_labels.append(img_path.split("/")[-1].split(".")[0].split("_")[1])
        
    
    train_embeddings = []
    train_labels = []
    for img_path in train_img_paths:
        represent_results = represent_face(img_path, select_model, select_backend)
        train_embeddings.append(represent_results[0]["embedding"])
        train_labels.append(img_path.split("/")[-1].split(".")[0].split("_")[1])

    # 유사도 계산
    eu_label_embedding_list = []
    for train_embedding in train_embeddings:
        eu_dist = [distance(train_embedding, test_embedding, similarity_algorithm) for test_embedding in test_embeddings]
        for pair in zip(eu_dist, test_labels):
            eu_label_embedding_list.append(pair)

    # 유사도 기준으로 정렬
    eu_label_embedding_list = sorted(eu_label_embedding_list)

    # 상위 10개 추출
    eu_top_rank_list = eu_label_embedding_list[:10]
    
    # 정확도 계산
    correct_count_dict = defaultdict(lambda : 0)
    for train_label in train_labels:
        count = calculate(eu_top_rank_list, train_label)
        correct_count_dict[train_label] += count
    
    st.write(correct_count_dict)
    
    # 정확도 그래프 생성
    item = correct_count_dict.items()
    item = sorted(item)
    name = []
    counts = []
    for label, count in item:
        name.append(label)
        counts.append(count)

    plt.figure(figsize=(25, 10))
    plt.bar(name, counts)
    st.pyplot(plt, use_container_width=True)

    
    

    
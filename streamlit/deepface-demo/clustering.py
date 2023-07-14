import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
import streamlit as st


def perform_kmeans_clustering(file_path):
    # pkl 파일 열기
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    # 임베딩 벡터 추출
    embeddings = [item[1] for item in data]
    
    # K-means 클러스터링
    kmeans = KMeans(n_clusters=5)  # 클러스터 개수 설정
    labels = kmeans.fit_predict(embeddings)

    # PCA를 사용하여 임베딩 벡터를 2차원으로 축소
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # 클러스터링 결과를 데이터프레임으로 변환
    df = pd.DataFrame({"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "label": labels})

    # 클러스터별 색상 설정
    colors = ["red", "blue", "green", "orange", "purple", "yellow", "cyan", "magenta", "pink", "gray"]

    # 시각화
    fig, ax = plt.subplots()
    for label in df["label"].unique():
        cluster_data = df[df["label"] == label]
        ax.scatter(cluster_data["x"], cluster_data["y"], c=colors[label % len(colors)], label=f"Cluster {label}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    # Streamlit에서 이미지로 출력
    st.pyplot(fig, use_container_width=True)


def perform_dbscan_clustering(file_path):
    # pkl 파일 열기
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # 임베딩 벡터 추출
    embeddings = [item[1] for item in data]

    # DBSCAN 클러스터링
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # eps, min_samples 설정
    labels = dbscan.fit_predict(embeddings)

    # PCA를 사용하여 임베딩 벡터를 2차원으로 축소
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # 클러스터링 결과를 데이터프레임으로 변환
    df = pd.DataFrame({"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "label": labels})
    
    # 시각화
    fig, ax = plt.subplots()
    for label in df["label"].unique():
        cluster_data = df[df["label"] == label]
        ax.scatter(cluster_data["x"], cluster_data["y"], label=f"Cluster {label}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    # Streamlit에서 이미지로 출력
    st.pyplot(fig, use_container_width=True)


def perform_hierarchical_clustering(file_path):
    # pkl 파일 열기
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # 임베딩 벡터 추출
    embeddings = [item[1] for item in data]

    # 계층적 클러스터링
    Z = linkage(embeddings, method="ward")  # linkage 메서드와 메서드에 따른 설정
    labels = fcluster(Z, t=5, criterion="maxclust")  # 클러스터 개수 설정

    # PCA를 사용하여 임베딩 벡터를 2차원으로 축소
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # 클러스터링 결과를 데이터프레임으로 변환
    df = pd.DataFrame({"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "label": labels})

    # 시각화
    fig, ax = plt.subplots()
    for label in df["label"].unique():
        cluster_data = df[df["label"] == label]
        ax.scatter(cluster_data["x"], cluster_data["y"], label=f"Cluster {label}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    # Streamlit에서 이미지로 출력
    st.pyplot(fig, use_container_width=True)
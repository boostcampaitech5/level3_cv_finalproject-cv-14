from torch.utils.data import Dataset, DataLoader
from Crawling_Dataset import Crawling_Nomal_Dataset
import torch
import cv2 
import os
import numpy as np
from torchvision import transforms
import torch.utils.data as data
from facenet_pytorch import MTCNN, fixed_image_standardization, InceptionResnetV1
import math
import tqdm as tqdm
# import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from torchvision.transforms import Resize
# from cheff import bring

from Embedding import Embedding_vector, Embeddings_Manager
from Label_DataFrame import Label_DataFrame
from sklearn.metrics import confusion_matrix

def calculate_mean_std(df) :
    p_mean = round(df[df.decision == "Yes"].distance.mean(), 4)
    p_std = round(df[df.decision == "Yes"].distance.std(), 4)
    n_mean = round(df[df.decision == "No"].distance.mean(), 4)
    n_std = round(df[df.decision == "No"].distance.std(), 4)
    print(p_mean, p_std)
    print(n_mean, n_std)
    return p_mean, p_std, n_mean, n_std

def get_threshold(p_mean, p_std, sigma=1) :
    threshold = round(p_mean + sigma * p_std, 4)
    return threshold

def get_statistic(df) :
    cm = confusion_matrix(df.decision.values, df.prediction.values)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn)/(tn + fp +  fn + tp)
    f1 = 2 * (precision * recall) / (precision + recall)
    print('acc    : ', accuracy)
    print('recall : ', recall)
    print('f1     : ', f1)
    print('precision : ', precision)
    return accuracy, recall, f1, precision


def validation(model, vali_data_loader, test_path) :

    model_vector= Embedding_vector(model=model)
    model_vector_imform = Embeddings_Manager(file_path=test_path, embedding_vector=model_vector, dataloader=vali_data_loader)
    model_identities = model_vector_imform.get_label_per_path_dict()
    model_path2embedding = model_vector_imform.get_path_embedding_dict()

    model_df = Label_DataFrame(identities=model_identities)
    positive_df = model_df.get_positive_df()
    negative_df = model_df.get_negative_df()
    facenet_label_df = model_df.concate()
    model_inference_df = model_df.get_inference_df(model_path2embedding)

    p_mean, p_std, n_mean, n_std = calculate_mean_std(model_inference_df)
    threshold = get_threshold(p_mean, p_std, sigma=1)
    facenet_prediction_df = model_df.get_prediction_df(threshold=threshold)

    accuracy, recall, f1, precision = get_statistic(facenet_prediction_df)

    return accuracy, recall, f1, precision

if __name__ == '__main__' :
    model_path = '/opt/ml/insightface/recognition/arcface_torch/work_dirs/wf4m_r50_epoch20/model.pt'
    train_path = '/opt/ml/data/celeb/cut_train'
    test_path = '/opt/ml/data/celeb/cut_test' 

    transform = transforms.Compose([
        np.float32,
        transforms.ToTensor(),  # range [0, 255] -> [0.0, 1.0]
        Resize((112, 112)),
        fixed_image_standardization
    ])
    facenet = InceptionResnetV1(classify=False, pretrained='vggface2')

    test_dataset = Crawling_Nomal_Dataset(test_path, transforms=transform)
    test_data_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=False)
    
    accuracy, recall, f1, precision = validation(model=facenet, vali_data_loader=test_data_loader, test_path=test_path)






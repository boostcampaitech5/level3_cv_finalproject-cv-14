from typing import Any
from torch.utils.data import Dataset, DataLoader
from Crawling_Dataset import Crawling_Nomal_Dataset
import torch
import cv2 
import os
import numpy as np
from torchvision import transforms
import torch.utils.data as data
import math
from PIL import Image
from collections import defaultdict


def img_loader(path) :
    try : 
        with open(path, 'rb') as f :
            img = cv2.imread(path) 
            if len(img.shape) == 2 :
                img = np.stack([img] * 3, 2)
        
            return img
    except IOError :
        print('Cannot load image' + path)

def evaluate_model(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        self.embedding_vector.model.eval()
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper

class Embedding_vector :
    def __init__(self, model, transform=None) :
        self.transform = transform
        self.model = model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def get_embedding(self, image_path=None, feature=None) :
        # image_path로 기입된 경우와 image자체로 기입된 경우를 나눈다.
        if image_path is not None :
            img = img_loader(image_path)
        else : 
            img = feature
        
        # if self.transform is not None : 
        #     img = self.transform(img)
        # else :
        #     img = torch.from_numpy(img)

        # dataset을 거치지 않고 나온 이미지는 3차원이기 때문에 모델에 넣어줄 수 있게 4차원 변환
        if len(img.shape) == 3:         
            img = img.unsqueeze(0)
         
        img = img.to(self.device)

        # 만약 model이 GPU연산이 안되어있다면,
        if not next(self.model.parameters()).is_cuda:  
            self.model.to(self.device)
            
        embedding = self.model(img)
        embedding = embedding.to('cpu').numpy()   # embedding 연산은 CPU로 진행하기 위해 변환

        return embedding
    

## TODO data_loader가있는 경우 transform이 되어있는데, 굳이 embedding_vector에서 정의할 필요가 없지 않나.. 이걸 구분해주는 코드 필요

class Embeddings_Manager :
    def __init__(self, file_path, embedding_vector: Embedding_vector, dataloader = None) :
        self.functions = []
        self.file_path = file_path
        self.embedding_vector = embedding_vector
        self.data_loader = dataloader

    def get_label_per_path_dict(self) :
        identities = defaultdict(lambda : list())
        # dict = {label1 : [image_path_1.jpg,image_path_2.jpg,....],
        #        label2 : [image_path_1.jpg,image_path_2.jpg,....],...}

        if self.data_loader is  None :    # dataloader를 특별히 지정하지 않은 경우
            dataset = Crawling_Nomal_Dataset(self.file_path, transforms=self.embedding_vector.transform)
            data_loader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=False)
        else :
            data_loader = self.data_loader

        for img,  path_label in data_loader :
            # print(img, path_label)
            img_path = path_label[0][0]
            img_label = path_label[1].tolist()[0]
            # print(img_path, img_label, sep='\n')
            identities[img_label].append(img_path)
        return identities

    @evaluate_model   # model.eval(), torch.no_grad() wrapping한 데코레이터
    def get_path_embedding_dict(self) :
        path_embedding_dict = {}

        if self.data_loader is  None :    # dataloader를 특별히 지정하지 않은 경우
            dataset = Crawling_Nomal_Dataset(self.file_path, transforms=self.embedding_vector.transform)
            data_loader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=False)
        else :
            data_loader = self.data_loader
        
        for feature, path_label_list in data_loader :
            # print(path_label_list)
            embeddings = self.embedding_vector.get_embedding(feature=feature)
            key = path_label_list[0][0]
            value = embeddings
            path_embedding_dict[key] = value

        return path_embedding_dict

        

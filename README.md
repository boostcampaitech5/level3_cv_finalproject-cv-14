# Memory sutdio
## Intro

<!--![](Appendix/intro.png)-->

>memory studio는 크게 두가지 폴더로 구성되어있습니다. facerecognition은 arcface loss를 이용한 학습과 pretrained model을 불러와서 finetuning하는데 초점을 두었습니다.
>그리고 service는 AWS에 서버를 두고 node.js와 monogoDB를 연결하며, v100서버에 flask를 통해 모델을 서빙하여 배포했습니다.

```
Face Recognition
├── Backbone
├── configs
├── docs
  ├── dataset.py
  ├── train.py
  ├── validation.py
  └── inference.py
    
Service
├── Dockerfile
├── Docker-compose.yml
├── requirements.txt
└── app
	├── main.py
	├── model.py
	├── dataset.py
	└── inference.py

└── input_images
    └── image.jpg
    
```


# Web Page URL
### http://memory-studio.ap-northeast-2.elasticbeanstalk.com/

## Member

![](Appendix/member.png)

|  이름  | github                          |
| :----: |  ------------------------------- |
| 김우진 |  https://github.com/w-jnn        |
| 신건희 |  https://github.com/Rigel0718    |
| 신중현 |  https://github.com/Blackeyes0u0 |
| 이종휘 |  https://github.com/gndldl       |


# Face Detection & Recognition

requirements.txt: 필요한 파이썬 라이브러리를 기술한 파일입니다.




####  Pretrained Models & Performance

| 모델 | 데이터 세트 | accuracy | recall | F1 score | precision |
|---|---|---|---|---|---|
| Arcface(Resnet 18) | MS1MV3 | 0.5485 | 0.6102 | 0.4579 | 0.3664 |
| Arcface(mobilenet) | Face emore | 0.5321 | 0.5906 | 0.4410 | 0.3519 |
| Facenet(Inception) | VGGface2 | 0.8810 | 0.8382 | 0.8262 | 0.8096 |


#### How to use

- clone

  ```
  git clone https://github.com/deepinsight/insightface.git
  ```


#### Prepare Dataset ( For training)

download the refined dataset: (emore recommended)

- [emore dataset @ BaiduDrive](https://pan.baidu.com/s/1eXohwNBHbbKXh5KHyItVhQ), [emore dataset @ Dropbox](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0)
- More Dataset please refer to the [original post](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)


------

### 3.2 dataset structure

```
- facedataset/
         name1/
             photo1.jpg
             photo2.jpg
             ...
         name2/
             photo1.jpg
             photo2.jpg
             ...
         .....
```

- - -

#

### 3.3 Training:

```bash
python train.py configs/config
```

## 4. References 

- This repo is mainly inspired by [deepinsight/insightface](https://github.com/deepinsight/insightface) and [FaceNet](https://github.com/davidsandberg/facenet)

---
## Service Architecture

![](Appendix/servicear.png)

>Dockerfile: 도커 이미지를 빌드하기 위한 파일입니다.
main.py: fastapi를 docker-compose 파일입니다.



![](Appendix/service.png)

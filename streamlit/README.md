# streamlit 사용 방법

## Installation

**streamlit**

```shell
$ pip install streamlit
```

```shell
$ pip install streamlit-option-menu
```

**deepface demo**

```shell
$ pip install deepface
```

**facenet demo**

```shell
$ pip install facenet_pytorch
```

## Run

```shell
$ streamlit run app.py --server.port {포트번호}
```

External URL 접속

# Contents

```
streamlit
├── .streamlit
│   └── config.toml
│
├── static
│   └── celeb
│       ├── cut_test
│       ├── cut_train
│       ├── test
│       └── train
│
├── deepface-demo
│   ├── app.py
│   ├── calculate.py
│   ├── clustering.py
│   ├── predict.py
│   └── utils.py
│
├── facenet-demo
│
└── README.md

```

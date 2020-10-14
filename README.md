


# Deeplab v3 plus for Landcover segmentation 

## 1. conda 혹은 virtualenv 설치
- virtualenv 설치 명령 :

   `virtualenv --system-site-packages -p python3 ./venv`
   
## 2. virtualenv 활성화 혹은 conda 활성화 후 파이썬 모듈 pip 설치
- virtualenv 활성화 :

    `source ./venv/bin/activate`
    
- pip 설치

    `cd Deeplab_allforland`
    
    `pip3 install -r requirements.txt`
## 3. 실행
- 데이터셋 폴더 구성 :
```bash
Landcover_dataset(root)
├── train
│   ├── image
│   │   └── image
│   └── mask_detail
│       └── mask
└── valid
    ├── image
    │   └── image
    └── mask
        └── mask
```
- 학습 진행(train.py) : 

  `python train.py --dataset_path='데이터셋 root 경로(train,valid폴더가 들어있는 Landcover_dataset폴더의 절대경로)'`
  
- 평가만 진행(val.py) :

  `python val.py --dataset_path='데이터셋 root 경로(train,valid폴더가 들어있는 Landcover_dataset폴더의 절대경로)'`
  
- 학습된 데이터로 추론 결과 이미지 확인 :

  `python demo.py --in_path='추론할 이미지가 들어있는 폴더 경로' --out_path='추론된 이미지가 저장되는 폴더 경로'`
  
## 4. 결과
- 경로 자동생성 :  Deeplab_allforland/run/deeplab-(backbone이름)/experiment_*
- 결과물 및 로그 : 학습파라미터 관련 로그, checkpoint.tar.pth, tensorboard 로그용 tfrecords


### Base code
- https://github.com/jfzhang95/pytorch-deeplab-xception.git

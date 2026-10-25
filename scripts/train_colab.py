# train.py (Colab용, dataset.zip 안에 train.py 이름으로 들어감)
# zip 구성: train.py + dataset/{data.yaml, train/, val/}
# Colab 셀에서:
#   from google.colab import drive; drive.mount('/content/drive')
#   !unzip -q /content/drive/MyDrive/dataset.zip -d /content
#   !pip install -q ultralytics
#   !python train.py
# 결과(pt 포함)는 구글 드라이브 MyDrive/yolo_runs/{NAME} 에 저장
# 세션이 끊기면 위 셀을 다시 실행하면 last.pt 에서 이어서 학습
import os
import re
import sys

import torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(HERE, 'dataset', 'data.yaml')
DRIVE = '/content/drive/MyDrive'
PROJECT = f'{DRIVE}/yolo_runs'
NAME = 'run_v1'  # 학습마다 이름 바꾸기 (같은 이름 폴더가 드라이브에 있으면 이어서 학습/종료 처리됨)

TRAIN_ARGS = dict(
    epochs=100,
    patience=20,
    imgsz=640,          # 원본 사진 416~640, 배포 모델도 640
    batch=32,           # T4 16GB 기준, 메모리 부족하면 16
    optimizer='AdamW',  # auto면 lr0가 무시되므로 명시
    lr0=0.001,
    device=0,
)


def main():
    if not os.path.isdir(DRIVE):
        sys.exit('구글 드라이브가 연결되지 않았습니다. 먼저 셀에서 실행하세요:\n'
                 "  from google.colab import drive; drive.mount('/content/drive')")
    if not torch.cuda.is_available():
        sys.exit('GPU가 없습니다. 런타임 > 런타임 유형 변경 > T4 GPU 선택 후 다시 실행하세요.')

    # data.yaml 의 path 를 압축 푼 위치로 맞춤
    with open(DATA_YAML) as f:
        text = f.read()
    text = re.sub(r'^path:.*$', f'path: {os.path.dirname(DATA_YAML)}', text, count=1, flags=re.M)
    with open(DATA_YAML, 'w') as f:
        f.write(text)

    last = f'{PROJECT}/{NAME}/weights/last.pt'
    if os.path.exists(last):
        # 학습이 끝나면 last.pt 의 epoch 이 -1 로 바뀜 (이 상태로 resume 하면 엉뚱한 기본 데이터로 학습됨)
        if torch.load(last, map_location='cpu', weights_only=False).get('epoch', -1) == -1:
            print(f'=== 학습이 이미 끝났습니다: {PROJECT}/{NAME}/weights/best.pt ===')
            print(f'새로 학습하려면 드라이브의 {PROJECT}/{NAME} 폴더 이름을 바꾸세요.')
            return
        print(f'=== 이어서 학습: {last} ===')
        YOLO(last).train(resume=True)
    else:
        print('=== 학습을 시작합니다 ===')
        YOLO('yolo11n.pt').train(data=DATA_YAML, project=PROJECT, name=NAME, exist_ok=True, **TRAIN_ARGS)
    print(f'=== 학습 완료: {PROJECT}/{NAME}/weights/best.pt ===')


if __name__ == '__main__':
    main()

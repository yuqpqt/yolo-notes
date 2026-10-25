[English](../en/04-colab-training-notes.md) | **한국어**

# 4. Colab 학습과 결과 측정

## 끊겨도 이어지는 설정

[`scripts/train_colab.py`](../../scripts/train_colab.py)는 학습 결과를 구글 드라이브에 저장하고, 끊기면 `last.pt`에서 이어서 학습한다.

zip 구성: `train.py`(이 스크립트 이름을 바꾼 것) + `dataset/{data.yaml, train/, val/}`.

```python
from google.colab import drive; drive.mount('/content/drive')
!unzip -q /content/drive/MyDrive/dataset.zip -d /content
!pip install -q ultralytics
!python train.py
```

끊기면 같은 셀을 다시 실행한다.

## 겪은 문제

| 문제 | 원인 | 해결 |
|---|---|---|
| 이어서 학습했더니 엉뚱한 작은 데이터(`coco8`)로 학습됨 | 학습이 끝나면 `last.pt`의 `epoch`이 -1이 되고, 이걸로 `resume=True` 하면 데이터 경로를 잃음 | `torch.load(last)['epoch'] == -1`이면 이어서 학습하지 않고 종료 |
| 스크립트 안에서 `drive.mount` 실패 | 노트북 셀에서만 동작 | 셀에서 마운트하고, 스크립트는 `/content/drive/MyDrive`가 있는지만 확인 |
| `data.yaml`을 못 찾음 | `path: .`이 실행 위치/Ultralytics 데이터 폴더 기준으로 해석됨 | 시작할 때 `path:`를 압축 푼 위치로 바꿈 |
| `BadZipFile` | 드라이브 업로드가 끝나기 전에 압축을 풂 | 업로드 완료 후 파일 크기 확인 |
| batch 0에서 멈춤 | 작은 Colab VM에서 `cache='ram'` | `cache` 옵션 제거 |
| `batch=64`로 올려도 안 빨라짐 | Ultralytics는 어차피 64장 단위로 기울기를 모아서 업데이트하고, 무료 Colab(CPU 2개)은 GPU가 아니라 데이터 로딩이 병목 | `batch=32` 유지, T4에서 약 4,500장 기준 에포크당 약 1분 |
| 가만히 두면 세션이 끊김 | 무료 Colab은 유휴 세션을 끊음 | 드라이브에서 자동 이어서 학습, 새 학습마다 `NAME` 변경 |
| `optimizer='auto'`면 `lr0`가 무시됨 | `auto`가 학습률을 직접 정함 | `optimizer='AdamW'`를 명시 |

사용한 설정: `epochs=100, patience=20, imgsz=640, batch=32, optimizer='AdamW', lr0=0.001`.

## 실제 영상으로 측정하기

val mAP는 모델이 *내 val 데이터*에 얼마나 맞는지만 알려준다.
첫 모델은 블랙박스 영상에서 포트홀·크랙을 거의 못 잡았고, 이후 모델(val mAP50 0.715)은 포트홀 138개를 잡았는데
확인한 121개가 전부 맨홀이었다. val 사진이 학습과 같은 공개 데이터에서 나왔기 때문에 두 문제 모두 val 점수로는 보이지 않았다.

대신 이렇게 측정한다.

- **학습에 쓰지 않는 고정 시험용 영상**
- **실제 사용할 임계값(conf 0.5)에서 클래스별 정확도**: 맞음 ÷ 검출 수.
  목표: 포트홀·크랙·차선 80% 이상, 데이터가 적은 시설물 클래스는 50% 정도.
- **긴 영상 빠르게 훑기**: `--sample-sec 1.5`로 1.5초마다 한 프레임으로 바로 이동하면,
  30fps 영상 약 160분을 GTX 1050 Ti에서 약 20분에 검출했다.
  (`--stride`는 모든 프레임을 디코딩해서 긴 영상에서는 더 느리다.)

## 학습 없이 학습된 모델에 클래스 추가하기

기존 프로그램은 8종을 기대하는데 내 모델은 7종이었다.
검출 헤드의 마지막 conv(`cv3[i][-1]`)에 출력 채널을 하나 추가하고 (가중치 0, bias −20),
`nc=8`과 이름을 바꿨다.
0~6번 결과는 완전히 같고, 7번은 절대 검출되지 않는다.
호환성만 맞추는 방법이라, 실제로 그 클래스를 잡으려면 학습해야 한다.

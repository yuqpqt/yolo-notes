[English](README.md) | **한국어**

# yolo-notes — YOLO11로 도로 결함 검출 모델을 학습하며 배운 것

한국 도로 **블랙박스 영상**에서 도로 결함(포트홀, 크랙, 차선도색불량, 도로 시설물 파손)을 찾는
YOLO11n 모델을 학습하면서 정리한 노트와 스크립트입니다.

처음 시작할 때 몰랐다가 직접 실패해 보고 알게 된 것들입니다.
교훈마다 그걸 알려준 실험 결과를 같이 적었습니다.

## 핵심 교훈

1. **모든 클래스를 매번 한 번에 학습한다.**
   잘 되던 모델에 새 클래스 데이터만 추가 학습했더니 기존 클래스를 잊어버렸다
   (포트홀 검출 87개 → 0개). 데이터셋을 master 하나로 모으고 매번 전체로 다시 학습한다.
   → [docs/ko/01-train-all-classes-together.md](docs/ko/01-train-all-classes-together.md)
2. **학습 사진은 실제로 쓸 카메라와 같은 시점이어야 한다.**
   위에서 내려다본 사진, 드론 사진, 정사각형으로 찌그러진 사진으로는 블랙박스 영상에서 거의 못 잡았다.
   → [docs/ko/02-dataset-collection-tips.md](docs/ko/02-dataset-collection-tips.md)
3. **Roboflow에서 받을 때 증강은 끄고, 뒤집어진 사진은 뺀다.**
   YOLO는 학습할 때 알아서 증강하고, 블랙박스는 도로를 거꾸로 보지 않는다.
4. **오탐 사진을 배경으로 다시 넣는다.**
   맨홀을 포트홀로 잡은 프레임을 배경으로 넣었더니 그 오탐이 사라졌다 (포트홀 오탐 138 → 0).
   → [docs/ko/03-hard-negative-mining.md](docs/ko/03-hard-negative-mining.md)
5. **val mAP가 목표가 아니다. 내 영상으로 측정한다.**
   val mAP50 0.715인 모델이 블랙박스 영상에서 포트홀 138개를 잡았는데, 확인한 121개가 전부 맨홀이었다.
   영상 하나는 학습에서 빼서 고정 시험용으로 쓴다.
   → [docs/ko/04-colab-training-notes.md](docs/ko/04-colab-training-notes.md) (Colab 주의사항 포함)

## 클래스

| id | 클래스 | 비고 |
|---|---|---|
| 0 | Pothole | 포트홀 |
| 1 | Crack | 가로·세로 크랙 통합 (거북등 크랙 제외) |
| 2 | Damaged_median | 중앙분리대 파손 |
| 3 | Damaged_guide post | 시선유도봉 파손 |
| 4 | Damaged_bollard | 볼라드 파손 |
| 5 | Damaged_traffic sign | 데이터 없음, 검출 대상 아님 |
| 6 | Poor_lane | 차선도색불량 |

## 지금까지 결과

블랙박스 영상 4개(약 160분, 1.5초마다 1장, conf 0.5) 검출 수:

| 클래스 | v3 | v5 (+ 오탐 배경 사진) |
|---|---|---|
| 포트홀 | 138 (확인한 121개 모두 맨홀) | 0 |
| 크랙 | 609 | 226 |
| 차선도색불량 | 132 | 67 |

val (v5 최고 에포크): precision 0.735, recall 0.695, mAP50 0.734, mAP50-95 0.468.
실제 영상에서 크랙·차선 정확도는 아직 측정 중입니다.

## 스크립트

| 파일 | 하는 일 |
|---|---|
| [scripts/train_colab.py](scripts/train_colab.py) | Colab 학습. 구글 드라이브에 저장하고 끊기면 이어서 학습 |
| [scripts/custom_detect.py](scripts/custom_detect.py) | 사진·영상·카메라 검출, 검출될 때마다 박스 사진·원본·YOLO 라벨 저장 |

### custom_detect.py 사용법

```bash
python custom_detect.py --model best.pt --source video.mp4 --resolution 1280x720 --no-gnss --thresh 0.5
```

| 옵션 | 설명 |
|---|---|
| `--model` | YOLO 모델 경로 (필수) |
| `--source` | 사진, 폴더, 영상, `usb0`(USB 카메라), `picamera0` (필수) |
| `--thresh` | 표시·저장할 최소 신뢰도 (기본 0.5) |
| `--resolution` | 출력·저장 해상도 `가로x세로` |
| `--device` | `cuda`(기본) 또는 `cpu` |
| `--record` | 검출 화면을 `demo1.avi`로 녹화 |
| `--stride N` | 영상: N프레임마다 1장 검출 |
| `--sample-sec S` | 영상: S초마다 1장으로 바로 이동해서 검출 (긴 영상 빠르게 훑기) |
| `--exclude` | 제외할 클래스 이름 또는 번호, 예: `"Poor_lane,6"` |
| `--gnss-port`, `--gnss-baud`, `--no-gnss` | GNSS(NMEA) 시리얼 위치 정보 표시 |

키: `q` 종료, `s` 일시정지, `p` 화면 저장.
검출 결과는 최대 1초에 1번 `result/`(박스 표시), `result_raw/`(원본), `result_label/`(YOLO txt)에
`{클래스}_{영상이름}_f{프레임번호}` 이름으로 저장됩니다.

## 포함하지 않은 것

블랙박스 영상과 프레임(번호판, 얼굴, 위치 정보), 외부 데이터셋 사진(라이선스는 각 Roboflow 페이지 참고),
모델 가중치(pt).

## 라이선스

[MIT](LICENSE)

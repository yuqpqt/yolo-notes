import os
import sys
import argparse
import glob
import time
import subprocess

import cv2
import numpy as np
from ultralytics import YOLO
import serial
import threading

from datetime import datetime

# time check init
last_saved_time = 0

def get_board_temp():
    # 라즈베리파이 'vcgencmd measure_temp' 명령으로 보드 온도를 읽는 함수
    # vcgencmd 명령이 없는 환경(라즈베리파이가 아닌 보드/PC 등)에서는 None을 반환
    try:
        output = subprocess.check_output(['vcgencmd', 'measure_temp']).decode('utf-8')
        temp_str = output.strip().split('=')[1].split("'")[0]
        return float(temp_str)
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError, ValueError):
        return None

# ----------------------------------------------------------------------------
# GNSS(NMEA) 수신 기능
# ----------------------------------------------------------------------------

gnss_lock = threading.Lock()
latest_gnss_sentence = None
latest_gnss_fix = None
gnss_thread_status = 'not started'


def nmea_coord_to_decimal(coord_str, direction):
    if not coord_str or not direction:
        return None
    try:
        dot_idx = coord_str.find('.')
        if dot_idx < 2:
            return None
        deg_len = dot_idx - 2
        degrees = float(coord_str[:deg_len])
        minutes = float(coord_str[deg_len:])
        decimal = degrees + (minutes / 60.0)
        if direction in ('S', 'W'):
            decimal = -decimal
        return decimal
    except (ValueError, IndexError):
        return None


def parse_nmea_gga(sentence):
    try:
        body = sentence.split('*')[0]
        parts = body.split(',')
        if len(parts) < 10:
            return None
        if not parts[0].endswith('GGA'):
            return None
        lat = nmea_coord_to_decimal(parts[2], parts[3])
        lon = nmea_coord_to_decimal(parts[4], parts[5])
        return {
            'time_utc': parts[1],
            'lat': lat,
            'lon': lon,
            'fix_quality': parts[6],
            'num_sats': parts[7],
            'altitude_m': parts[9],
        }
    except (ValueError, IndexError):
        return None


def open_gnss_uart(port, baudrate):
    return serial.Serial(
        port=port,
        baudrate=baudrate,
        timeout=1.0,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
    )


def gnss_reader_thread(port, baudrate, retry_interval_sec=3.0):
    global latest_gnss_sentence, latest_gnss_fix, gnss_thread_status

    ser = None
    while ser is None:
        try:
            ser = open_gnss_uart(port, baudrate)
        except serial.SerialException as e:
            with gnss_lock:
                gnss_thread_status = 'error'
            print(f"GNSS ERROR: serial port open failed ({port}, {baudrate}bps): {e}")
            print(f"GNSS: {retry_interval_sec:.0f}초 후 재시도합니다...")
            time.sleep(retry_interval_sec)

    with gnss_lock:
        gnss_thread_status = 'running'
    print(f"GNSS: Listening on {port} at {baudrate} bps...")

    while True:
        try:
            if ser.in_waiting:
                data = ser.readline()
                if not data:
                    continue
                raw_line = data.decode(errors='replace').strip()
                if raw_line:
                    print(f"GNSS Received: {raw_line}")
            else:
                time.sleep(0.05)
                continue
        except serial.SerialException as e:
            with gnss_lock:
                gnss_thread_status = 'error'
            print(f"GNSS ERROR: serial read failed, reconnecting: {e}")
            try:
                ser.close()
            except Exception:
                pass
            time.sleep(retry_interval_sec)
            try:
                ser = open_gnss_uart(port, baudrate)
                with gnss_lock:
                    gnss_thread_status = 'running'
                print(f"GNSS: reconnected on {port} at {baudrate} bps")
            except serial.SerialException:
                pass
            continue
        except Exception:
            continue

        if not raw_line:
            continue

        if not raw_line.startswith('$'):
            continue

        parsed = parse_nmea_gga(raw_line)
        with gnss_lock:
            latest_gnss_sentence = raw_line
            gnss_thread_status = 'running'
            if parsed is not None:
                latest_gnss_fix = parsed


# save folder check & automatic creation
save_dir = './result'
save_raw_dir = './result_raw'
save_label_dir = './result_label'

for folder_path in [save_dir, save_raw_dir, save_label_dir]:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"info: '{folder_path}' folder generates")

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")', required=True)
parser.add_argument('--source', help='Image source, can be image file, folder, video, USB camera, or Pi Camera', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects', default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at', default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi".', action='store_true')
parser.add_argument('--device', help='Device to run inference on (cuda or cpu)', default='cuda') 
parser.add_argument('--gnss-port', help='Serial port for GNSS(NMEA) input', default='/dev/ttyAMA2')
parser.add_argument('--gnss-baud', help='Baudrate for GNSS serial port', type=int, default=115200)
parser.add_argument('--no-gnss', help='Disable GNSS(NMEA) serial reading', action='store_true')
parser.add_argument('--stride', help='Video only: run detection on every N-th frame (e.g. 6 = 5 frames/sec for 30fps video)', type=int, default=1)
parser.add_argument('--sample-sec', help='Video only: jump (seek) to one frame every N seconds of video, for quick scans of long videos (overrides --stride)', type=float, default=0)
parser.add_argument('--exclude', help='Comma-separated class names or ids to ignore (e.g. "Poor_lane" or "6")', default='')
args = parser.parse_args()


model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record
inference_device = args.device
gnss_port = args.gnss_port
gnss_baud = args.gnss_baud
gnss_enabled = not args.no_gnss
frame_stride = max(1, args.stride)

if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

model = YOLO(model_path, task='detect')
labels = model.names

# --exclude 로 지정한 클래스는 검출에서 제외
exclude_set = {x.strip() for x in args.exclude.split(',') if x.strip()}
keep_classes = [i for i, n in labels.items() if str(i) not in exclude_set and n not in exclude_set]
if exclude_set:
    print(f'Excluded classes: {[n for i, n in labels.items() if i not in keep_classes]}')

if gnss_enabled:
    gnss_thread = threading.Thread(
        target=gnss_reader_thread,
        args=(gnss_port, gnss_baud),
        daemon=True
    )
    gnss_thread.start()
else:
    gnss_thread_status = 'disabled'

img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif img_source.startswith('usb'):
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif img_source.startswith('picamera'):
    source_type = 'picamera'
    picam_idx_str = img_source[8:]
    picam_idx = int(picam_idx_str) if picam_idx_str != '' else 0
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)


resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
elif source_type == 'picamera':
    resW, resH = 640, 480

if record:
    if source_type not in ['video','usb','picamera']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res and source_type != 'picamera':
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # 영상 시간 기준 저장 간격 / 파일명에 사용
    if source_type == 'video':
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        video_stem = os.path.splitext(os.path.basename(img_source))[0]
        frame_idx = -1
        last_saved_video_time = -1.0
        sample_step = max(1, round(args.sample_sec * video_fps)) if args.sample_sec > 0 else 0

    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Begin inference loop
while True:

    t_start = time.perf_counter()
    current_real_time = time.time()

    if source_type == 'image' or source_type == 'folder': 
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video' and sample_step:
        # N초마다 1장: 사이 프레임은 디코딩하지 않고 해당 위치로 바로 이동
        frame_idx = 0 if frame_idx < 0 else frame_idx + sample_step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break

    elif source_type == 'video':
        # stride-1 프레임은 디코딩만 하고 건너뜀
        ret = True
        for _ in range(frame_stride - 1):
            ret = cap.grab()
            if not ret:
                break
            frame_idx += 1
        if ret:
            ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
        frame_idx += 1
    
    elif source_type == 'usb': 
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break
            
    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if (frame is None):
            print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
            break
            
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))
        
    raw_frame = frame.copy() 
    img_h, img_w = frame.shape[:2]
    
    results = model(frame, device=inference_device, verbose=False, classes=keep_classes)

    detections = results[0].boxes
    object_count = 0
    person_detected_in_frame = False
    valid_detections_yolo_fmt = []
    best_conf = -1.0
    best_classname = None

    for i in range(len(detections)):

        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        # 가변 thresh 적용
        if conf > min_thresh:
            person_detected_in_frame = True
            if conf > best_conf:
                best_conf = conf
                best_classname = classname
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            object_count = object_count + 1

            # YOLO 표준 정규화 라벨 데이터 생성 (<class_id> <x_center> <y_center> <width> <height>)
            x_center = ((xmin + xmax) / 2.0) / img_w
            y_center = ((ymin + ymax) / 2.0) / img_h
            norm_w = (xmax - xmin) / img_w
            norm_h = (ymax - ymin) / img_h
            valid_detections_yolo_fmt.append(f"{classidx} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    # 이미지 및 라벨 일괄 저장 로직
    # 영상: 영상 시간 기준 1초 간격 / 그 외: 실제 시간 기준 1초 간격
    if source_type == 'video':
        video_time = frame_idx / video_fps
        save_due = video_time - last_saved_video_time >= 1.0
    else:
        save_due = current_real_time - last_saved_time >= 1.0

    if person_detected_in_frame and save_due:
        if source_type == 'video':
            # 같은 초에 여러 장 저장돼도 덮어쓰지 않도록 영상명+프레임번호 사용
            base_filename = f"{best_classname}_{video_stem}_f{frame_idx:06d}"
            last_saved_video_time = video_time
        else:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{best_classname}_{now}"
        
        # 1. 바운딩 박스 포함 이미지 저장 (result)
        save_path = os.path.join(save_dir, f"{base_filename}.jpg")
        cv2.imwrite(save_path, frame)
        
        # 2. 원본 이미지 저장 (result_raw)
        save_raw_path = os.path.join(save_raw_dir, f"{base_filename}.jpg")
        cv2.imwrite(save_raw_path, raw_frame)

        # 3. 라벨 텍스트 파일 저장 (result_label)
        save_label_path = os.path.join(save_label_dir, f"{base_filename}.txt")
        with open(save_label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(valid_detections_yolo_fmt) + '\n')
        
        print(f"Image & Label save completed: {base_filename} (result, result_raw, result_label)")

        person_detected_in_frame = False 
        last_saved_time = current_real_time

    if source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) 

        board_temp = get_board_temp()
        temp_str = f'{board_temp:.1f}C' if board_temp is not None else 'N/A'
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, f'Temp: {temp_str}  Time: {now_str}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    
    cv2.putText(frame, f'Number of objects: {object_count}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) 

    with gnss_lock:
        gnss_status_snapshot = gnss_thread_status
        gnss_sentence_snapshot = latest_gnss_sentence
        gnss_fix_snapshot = latest_gnss_fix

    if gnss_status_snapshot == 'disabled':
        gnss_text = 'GNSS: disabled'
    elif gnss_status_snapshot == 'error':
        gnss_text = 'GNSS: port error'
    elif gnss_fix_snapshot is not None and gnss_fix_snapshot.get('lat') is not None and gnss_fix_snapshot.get('lon') is not None:
        gnss_text = (
            f"GNSS: {gnss_fix_snapshot['lat']:.6f}, {gnss_fix_snapshot['lon']:.6f}  "
            f"Fix:{gnss_fix_snapshot['fix_quality']} Sat:{gnss_fix_snapshot['num_sats']}"
        )
    elif gnss_sentence_snapshot is not None:
        gnss_text = f'GNSS(raw): {gnss_sentence_snapshot}'
    else:
        gnss_text = 'GNSS: waiting for data...'

    cv2.putText(frame, gnss_text, (10,80), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2) 

    cv2.imshow('YOLO detection results',frame) 
    if record: recorder.write(frame)

    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): 
        break
    elif key == ord('s') or key == ord('S'): 
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): 
        cv2.imwrite('capture.png',frame)
    
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    avg_frame_rate = np.mean(frame_rate_buffer)


print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import os
import time
from datetime import datetime

# -----------------------------
# 설정
# -----------------------------
WIN_NAME = "Live Filter App (1:Color  2:Motion  3:Watermark)"
SAVE_DIR = "./09_realtime_camerafilter_app/result"
os.makedirs(SAVE_DIR, exist_ok=True)

# 파라미터
COLOR_RANGE = {  # 기본: 초록 계열
    "lower": np.array([40, 70, 70]),   # H,S,V
    "upper": np.array([80, 255, 255])
}
MORPH_KERNEL = np.ones((5, 5), np.uint8)
THRESH_MOTION = 30         # 모션 이진화 임계값
MIN_AREA = 600             # 너무 작은 컨투어 무시

# -----------------------------
# 유틸
# -----------------------------
def put_hud(img, text, rec=False):
    h, w = img.shape[:2]
    bar_h = 36
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
    if rec:
        cv2.circle(img, (w-20, 18), 8, (0,0,255), -1)

def mini_overlay(dst, small, x=10, y=50):
    sh, sw = small.shape[:2]
    if y+sh <= dst.shape[0] and x+sw <= dst.shape[1]:
        dst[y:y+sh, x:x+sw] = small

def color_tracking(frame):
    """HSV 색상 범위로 마스크 → 컨투어 표시"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_RANGE["lower"], COLOR_RANGE["upper"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, MORPH_KERNEL, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            cv2.circle(frame, (cx, cy), 6, (0,0,255), -1)
            cv2.putText(frame, f"({cx},{cy})", (cx+8, cy-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

    # 좌상단에 마스크 미니뷰
    mini = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (frame.shape[1]//5, frame.shape[0]//5))
    mini_overlay(frame, mini)
    return frame

def motion_detect(frame, prev_gray):
    """프레임 차이 기반 모션 마스크 → 컨투어 표시, prev_gray 갱신 반환"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    _, mask = cv2.threshold(diff, THRESH_MOTION, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  MORPH_KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, MORPH_KERNEL, iterations=2)

    motion_detected = False  # <- 추가

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        motion_detected = True   # <- 움직임 있으면 True
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            cv2.circle(frame, (cx, cy), 6, (0,0,255), -1)

    # 미니뷰
    mini = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                      (frame.shape[1]//5, frame.shape[0]//5))
    mini_overlay(frame, mini)

    return frame, gray, motion_detected


def watermark(frame, text=None):
    """반투명 바 + 텍스트/시간 워터마크"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    bar_h = 42
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
    stamp = text or "OpenCV Live / " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, stamp, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255,255,255), 2, cv2.LINE_AA)
    return frame

# -----------------------------
# 카메라 열기
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows면 DSHOW가 종종 안정적
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

# 원하는 해상도 힌트(지원 안 되면 무시)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# FPS 추정 (없으면 30으로)
cap_fps = cap.get(cv2.CAP_PROP_FPS)
if not cap_fps or cap_fps <= 1:
    cap_fps = 30.0

# 모션용 첫 프레임(grayscale)
ret, first = cap.read()
if not ret:
    cap.release()
    raise RuntimeError("첫 프레임을 읽지 못했습니다.")
prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

# 창
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

# 녹화기
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = None
recording = False

mode = 1  # 1: 색상 추적, 2: 모션, 3: 워터마크
last_saved = 0.0

last_motion_log = 0.0  # <- 추가

print("[키] 1:색상추적  2:모션감지  3:워터마크  |  s:스냅샷  r:녹화  c:모션배경리셋  q/ESC:종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if mode == 1:
        out = color_tracking(frame.copy())
        put_hud(out, "Mode 1: Color Tracking (HSV)", rec=recording)

    elif mode == 2:
        out, prev_gray, motion_flag = motion_detect(frame.copy(), prev_gray)
        put_hud(out, "Mode 2: Motion Detection  (c: reset bg)", rec=recording)

        # 1초마다 모션 감지 로그
        if motion_flag:
            now = time.time()
            if now - last_motion_log >= 1.0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 모션 감지!")
                last_motion_log = now

    else:  # mode == 3
        out = watermark(frame.copy())
        put_hud(out, "Mode 3: Watermark", rec=recording)

    cv2.imshow(WIN_NAME, out)

    # 녹화
    if recording:
        if writer is None:
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(os.path.join(SAVE_DIR, "live_record.mp4"),
                                     fourcc, cap_fps, (w, h))
        writer.write(out)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key == ord('1'):
        mode = 1
    elif key == ord('2'):
        mode = 2
    elif key == ord('3'):
        mode = 3
    elif key == ord('c') and mode == 2:
        prev_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)  # 현 프레임 기준 재설정
        print("배경 리셋")
    elif key == ord('s'):
        # 너무 자주 누르면 파일 충돌 방지
        ts = time.time()
        if ts - last_saved > 0.2:
            path = os.path.join(SAVE_DIR, f"snap_{int(ts)}.png")
            cv2.imwrite(path, out)
            print("스냅샷 저장:", path)
            last_saved = ts
    elif key == ord('r'):
        recording = not recording
        if not recording and writer is not None:
            writer.release()
            writer = None
            print("녹화 종료")
        elif recording:
            print(f"녹화 시작 @ {cap_fps:.1f}fps")

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()

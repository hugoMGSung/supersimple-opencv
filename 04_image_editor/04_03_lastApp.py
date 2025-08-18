import cv2
import numpy as np

# ---------------------------
# 1) 이미지 불러오기
# ---------------------------
img = cv2.imread('./03_theory/data/cat02.jpg')
if img is None:
    raise FileNotFoundError('이미지를 불러올 수 없습니다.')
original = img.copy()

# ---------------------------
# 3) 이미지 변경 update함수 
# ---------------------------
def update(_=None):
    alpha = cv2.getTrackbarPos("Contrast", "Editor") / 50  # 대비
    beta = cv2.getTrackbarPos("Brightness", "Editor") - 100  # 밝기
    blur_val = cv2.getTrackbarPos("Blur", "Editor")
    edge_val = cv2.getTrackbarPos("Edge", "Editor")

    # 대비·밝기 조절
    edited = cv2.convertScaleAbs(original, alpha=alpha, beta=beta)

    # 블러
    if blur_val > 0:
        k = blur_val * 2 + 1  # 홀수 커널
        edited = cv2.GaussianBlur(edited, (k, k), 0)

    # 엣지
    if edge_val > 0:
        edited = cv2.Canny(edited, edge_val, edge_val*2)
        edited = cv2.cvtColor(edited, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Editor", edited)

# ---------------------------
# 2) 윈도우 & 트랙바 생성
# ---------------------------
cv2.namedWindow('Editor')
cv2.createTrackbar('Contrast', 'Editor', 50, 150, update)   # 대비
cv2.createTrackbar('Brightness', 'Editor', 100, 200, update) # 밝기
cv2.createTrackbar('Blur', 'Editor', 0, 10, update)        # 블러 강도
cv2.createTrackbar('Edge', 'Editor', 0, 200, update)       # 엣지 민감도

update()

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):  # ROI 선택
        roi = cv2.selectROI("Editor", original, False)
        if roi != (0,0,0,0):
            x,y,w,h = roi
            roi_img = original[y:y+h, x:x+w]
            cv2.imshow("ROI", roi_img)

    elif key == ord('q'): # 종료
        break
    else:
        cv2.imshow('SuperSimple ImageEditor', original)

cv2.destroyAllWindows()
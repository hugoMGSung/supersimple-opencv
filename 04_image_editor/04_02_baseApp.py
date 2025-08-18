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
    pass

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
        cv2.imshow('SuperSimple ImageEditor', original)
        pass

    elif key == ord('q'): # 종료
        break
    else:
        cv2.imshow('SuperSimple ImageEditor', original)

cv2.destroyAllWindows()
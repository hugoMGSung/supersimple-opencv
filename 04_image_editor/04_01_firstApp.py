import cv2
import numpy as np

# ---------------------------
# 1) 이미지 불러오기
# ---------------------------
img = cv2.imread('./03_theory/data/cat02.jpg')
if img is None:
    raise FileNotFoundError('이미지를 불러올 수 없습니다.')
original = img.copy()

cv2.imshow('SuperSimple ImageEditor', original)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np

img1 = cv2.imread(r'images2/1.jpg', 1)
img2 = cv2.imread(r'images2/2.jpg', 1)

lab_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
lab_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

# тюльпаны
lab_img1[:, :, 0] = (lab_img1[:, :, 0].astype(float)*1.15).clip(0, 255).astype(np.uint8)
lab_img1[:, :, 1] = (lab_img1[:, :, 1].astype(float)*0.85).clip(0, 255).astype(np.uint8)
lab_img1[:, :, 2] = (lab_img1[:, :, 2].astype(float)*1.1).clip(0, 255).astype(np.uint8)

# сиреневые цветы
lab_img2[:, :, 0] = (lab_img2[:, :, 0].astype(float)*1.1).clip(0, 255).astype(np.uint8)
lab_img2[:, :, 1] = (lab_img2[:, :, 1].astype(float)*1.15).clip(0, 255).astype(np.uint8)
lab_img2[:, :, 2] = (lab_img2[:, :, 2].astype(float)*0.87).clip(0, 255).astype(np.uint8)

cv2.imwrite(r'images2\new_1.jpg', cv2.cvtColor(lab_img1, cv2.COLOR_Lab2BGR))
cv2.imwrite(r'images2\new_2.jpg', cv2.cvtColor(lab_img2, cv2.COLOR_Lab2BGR))

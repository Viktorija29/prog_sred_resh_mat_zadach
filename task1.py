import cv2
from math import log
import matplotlib.pyplot as plt

list_names = []
list_numbs = []

for n in range(-10, 11):
    list_names.append(n)
    s = f"{n}.jpg"
    img = cv2.imread(fr'images\{s}', cv2.IMREAD_GRAYSCALE)
    crop_img = img[400:464, 400:464]
#     cv2.imshow('sss', crop_img)
#     cv2.waitKey(0)
    num = 0
    for i in range(0, 64):
        for j in range(0, 64):
            num += crop_img[i, j]
    num = log(num/4096)
    list_numbs.append(num)

fig = plt.subplots()
plt.plot(list_names, list_numbs)
plt.grid()
plt.xlabel('EV')
plt.ylabel('Динамический диапазон')
plt.scatter(list_names, list_numbs, color='orange', s=10, marker='o')
plt.show()
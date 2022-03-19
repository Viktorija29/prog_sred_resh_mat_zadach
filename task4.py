import cv2
import numpy as np

height = 6

A = cv2.imread(r'images4/plaz.jpg')
B = cv2.imread(r'images4/palmi.jpg')
A = cv2.resize(A, (512, 512))
B = cv2.resize(B, (512, 512))

# пирамида Гаусса для A
GaussPir = A.copy()
gaussA = [GaussPir]
for i in range(height):
    # cv2.imshow("1", G)
    # cv2.waitKey(0)
    GaussPir = cv2.pyrDown(GaussPir)
    gaussA.append(GaussPir)

# пирамида Гаусса для B
GaussPir = B.copy()
gaussB = [GaussPir]
for i in range(height):
    GaussPir = cv2.pyrDown(GaussPir)
    gaussB.append(GaussPir)

# пирамида Лапласа для A
laplasA = [gaussA[height - 1]]
for i in range(height - 1, 0, -1):
    GE = cv2.pyrUp(gaussA[i])
    rows, cols, _ = gaussA[i - 1].shape
    GE = GE[:rows, :cols]
    L = cv2.subtract(gaussA[i - 1], GE)
    laplasA.append(L)

# пирамида Лапласа для B
laplasB = [gaussB[height - 1]]
for i in range(height - 1, 0, -1):
    GE = cv2.pyrUp(gaussB[i])
    rows, cols, _ = gaussB[i - 1].shape
    GE = GE[:rows, :cols]
    L = cv2.subtract(gaussB[i - 1], GE)
    laplasB.append(L)

# соединение левой и правой половин фотографий
LS = []
for la, lb in zip(laplasA, laplasB):
    _, cols, _ = la.shape
    ls = np.hstack((la[:, :int(cols / 2)], lb[:, int(cols / 2):]))
    LS.append(ls)

ls_ = LS[0]
for i in range(1, height):
    ls_ = cv2.pyrUp(ls_)
    rows, cols, _ = LS[i].shape
    ls_ = ls_[:rows, :cols]
    ls_ = cv2.add(ls_, LS[i])

# прямое объединение фотографий
direct = np.hstack((A[:, :int(cols / 2)], B[:, int(cols / 2):]))

cv2.imwrite(r'images4/with_piramid.jpg', ls_)
cv2.imwrite(r'images4/direct.jpg', direct)


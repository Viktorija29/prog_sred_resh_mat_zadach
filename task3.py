import cv2

# -------- Выравнивание гистограммы -------- #

image1 = cv2.imread(r'images3/1.jpg')
image_lab1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
image_lab1[:, :, 0] = cv2.equalizeHist(image_lab1[:, :, 0])
res1 = cv2.cvtColor(image_lab1, cv2.COLOR_LAB2BGR)
cv2.imwrite(r'images3/1_hist.jpg', res1)

image2 = cv2.imread(r'images3/2.jpg')
image_lab2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)
image_lab2[:, :, 0] = cv2.equalizeHist(image_lab2[:, :, 0])
res2 = cv2.cvtColor(image_lab2, cv2.COLOR_LAB2BGR)
cv2.imwrite(r'images3/2_hist.jpg', res2)

# -------- Локальное выравнивание гистограммы -------- #

imageLocal1 = cv2.imread(r'images3/1.jpg')
image_lab_loc1 = cv2.cvtColor(imageLocal1, cv2.COLOR_BGR2LAB)
clahe1 = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(16, 16))
image_lab_loc1[:, :, 0] = clahe1.apply(image_lab_loc1[:, :, 0])
res_loc1 = cv2.cvtColor(image_lab_loc1, cv2.COLOR_LAB2BGR)
cv2.imwrite(r'images3/1_loc_hist.jpg', res_loc1)

imageLocal2 = cv2.imread(r'images3/2.jpg')
image_lab_loc2 = cv2.cvtColor(imageLocal2, cv2.COLOR_BGR2LAB)
clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
image_lab_loc2[:, :, 0] = clahe2.apply(image_lab_loc2[:, :, 0])
res_loc2 = cv2.cvtColor(image_lab_loc2, cv2.COLOR_LAB2BGR)
cv2.imwrite(r'images3/2_loc_hist.jpg', res_loc2)

# -------- Гауссовское размытие -------- #

imageGauss2 = cv2.imread(r'images3/2.jpg')
imageGauss3 = cv2.imread(r'images3/3.jpg')
resGauss = cv2.GaussianBlur(imageGauss2, (15, 15), sigmaX=0)
resGaussX = cv2.GaussianBlur(imageGauss3, (101, 1), sigmaX=0)
resGaussY = cv2.GaussianBlur(imageGauss3, (1, 101), sigmaX=0)
cv2.imwrite(r'images3/2_Gauss.jpg', resGauss)
cv2.imwrite(r'images3/3_GaussX.jpg', resGaussX)
cv2.imwrite(r'images3/3_GaussY.jpg', resGaussY)

# -------- Фильтр Собеля -------- #
imageSobel= cv2.imread(r'images3/2.jpg')
dstx = cv2.Sobel(imageSobel, -1, 1, 0, ksize=3)
dsty = cv2.Sobel(imageSobel, -1, 0, 1, ksize=3)
cv2.imwrite(r'images3/2_Sobel_x.jpg', dstx)
cv2.imwrite(r'images3/2_Sobel_y.jpg', dsty)

# -------- Фильтр Лапласса -------- #
imageLaplass = cv2.imread(r'images3/2.jpg')
resLaplass = cv2.Laplacian(imageLaplass, cv2.CV_64F, ksize=3)
cv2.imwrite(r'images3/2_Laplass.jpg', resLaplass)

import cv2


def find_faces_haar(image_path):
    image = cv2.imread(image_path)
    # перевод в черно-белое для лучшего обнаружения
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # загрузка классификатора Хаара
    haar_class = cv2.CascadeClassifier(r'C:\Users\vikab\PycharmProjects'
                                            r'\prog_sred_resh_mat_zadach\venv\Lib'
                                            r'\site-packages\cv2\data'
                                            r'\haarcascade_frontalface_alt.xml')
    # вызов функции распознавания и получения массивов - координаты и размеры лица
    faces = haar_class.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5)
    print('Обнаружено лиц:', len(faces))
    # отрисовка прямоугольников вокруг обнаруженных лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # сохранение фотографии с обнаруженными лицами
    name = image_path.split('.')
    cv2.imwrite(name[0] + '_haar.' + name[1], image)


find_faces_haar('images5/foto1.jpg')
find_faces_haar('images5/foto2.jpg')
find_faces_haar('images5/foto3.jpg')

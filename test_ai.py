import cv2
import os
import numpy as np

# 获取当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []

    for dir_name in dirs:
        if not dir_name.isdigit():
            continue

        label = int(dir_name)
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue

            image_path = os.path.join(subject_dir_path, image_name)
            image = cv2.imread(image_path)

            face, rect = detect_face(image)
            if face is not None:
                faces.append(face)
                labels.append(label)

    return faces, labels

def train_model(faces, labels):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    return recognizer

def save_model(recognizer, model_path):
    recognizer.save(model_path)

def main():
    data_folder_path = os.path.join(current_dir, 'training_data')
    model_path = os.path.join(current_dir, 'trained_model2.yml')

    faces, labels = prepare_training_data(data_folder_path)
    recognizer = train_model(faces, labels)
    save_model(recognizer, model_path)
    print("模型训练完成并已保存到", model_path)

if __name__ == "__main__":
    main()
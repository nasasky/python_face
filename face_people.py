import cv2
import numpy as np
import os
import json

# 获取当前工作目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# 加载人脸识别模型
recognizer = cv2.face.LBPHFaceRecognizer_create()

def detect_face(img):
    if img is None:
        raise ValueError("图像为空")

    # 检查图像是否为彩色图像
    if len(img.shape) == 2 or img.shape[2] == 1:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 直方图均衡化
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        return None, None
    
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise ValueError(f"图像文件路径不存在: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    face, rect = detect_face(image)
    return face

def save_data_to_json(faces, labels, file_path):
    data = {'faces': [face.tolist() for face in faces], 'labels': labels}
    with open(file_path, 'w') as f:
        json.dump(data, f)

def load_data_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    faces = [np.array(face) for face in data['faces']]
    labels = data['labels']
    return faces, labels

def prepare_training_data(data_folder_path):
    faces = []
    labels = []
    
    for label_dir in os.listdir(data_folder_path):
        label_path = os.path.join(data_folder_path, label_dir)
        if not os.path.isdir(label_path):
            continue
        
        label = int(label_dir)
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            face = preprocess_image(image_path)
            if face is not None:
                faces.append(face)
                labels.append(label)
    
    return faces, labels

def train_model(data_folder_path, json_file_path):
    faces, labels = prepare_training_data(data_folder_path)
    if len(faces) == 0:
        raise ValueError("没有检测到任何人脸，无法训练模型")

    recognizer.train(faces, np.array(labels))
    recognizer.save('trained_model.yml')
    save_data_to_json(faces, labels, json_file_path)

def load_model(model_path):
    if not os.path.exists(model_path):
        raise ValueError(f"模型文件路径不存在: {model_path}")

    recognizer.read(model_path)

def compare_faces(img1_path, img2_path):
    face1 = preprocess_image(img1_path)
    face2 = preprocess_image(img2_path)

    if face1 is None:
        print(f"无法检测到人脸: {img1_path}")
        return
    if face2 is None:
        print(f"无法检测到人脸: {img2_path}")
        return

    label, confidence = recognizer.predict(face2)
    print(f"预测标签：{label}, 置信度：{confidence}")

    if label == 0:
        print(f"人脸匹配，置信度：{confidence}")
    else:
        print("人脸不匹配")

# 示例使用
data_folder_path = os.path.join(current_dir, 'training_data')
json_file_path = os.path.join(current_dir, 'face_data.json')
try:
    train_model(data_folder_path, json_file_path)
    load_model('trained_model.yml')
    compare_faces(os.path.join(current_dir, 'face1.jpeg'), os.path.join(current_dir, 'face7.jpeg'))
except Exception as e:
    print(f"发生错误: {e}")
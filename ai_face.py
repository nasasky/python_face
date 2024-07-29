import cv2
import numpy as np
import os

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
    return gray[y:y + w, x:x + h], faces[0]


def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise ValueError(f"图像文件路径不存在: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    face, rect = detect_face(image)
    return face


def load_model(model_path):
    if not os.path.exists(model_path):
        raise ValueError(f"模型文件路径不存在: {model_path}")

    recognizer.read(model_path)


def recognize_face(image_path):
    face = preprocess_image(image_path)

    if face is None:
        print(f"无法检测到人脸: {image_path}")
        return

    label, confidence = recognizer.predict(face)
    print(f"预测标签：{label}, 置信度：{confidence}")

    if confidence < 100:  # 置信度阈值可以根据需要调整
        print(f"识别成功，标签：{label}, 置信度：{confidence}")
    else:
        print("识别失败")


# 示例使用
model_path = os.path.join(current_dir, 'trained_model.yml')
image_path = os.path.join(current_dir, 'face3.jpeg')

try:
    load_model(model_path)
    recognize_face(image_path)
except Exception as e:
    print(f"发生错误: {e}")
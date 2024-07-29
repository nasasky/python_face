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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces, gray

def load_model(model_path):
    if not os.path.exists(model_path):
        raise ValueError(f"模型文件路径不存在: {model_path}")
    recognizer.read(model_path)

def recognize_face(face, gray):
    label, confidence = recognizer.predict(face)
    return label, confidence

def main():
    model_path = os.path.join(current_dir, 'trained_model.yml')
    try:
        load_model(model_path)
    except Exception as e:
        print(f"发生错误: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            break

        faces, gray = detect_face(frame)
        for (x, y, w, h) in faces:
            face = gray[y:y+w, x:x+h]
            label, confidence = recognize_face(face, gray)
            if confidence < 100:  # 置信度阈值可以根据需要调整
                text = f"ID: {label}, Conf: {confidence:.2f}"
            else:
                text = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
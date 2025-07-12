import cv2
import numpy as np

# 1. 加载 Haar 级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. 初始化 LBF 面部关键点检测器
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")  # 确保模型文件路径正确

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸并转换为 NumPy 数组
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    faces_np = np.array([[x, y, w, h] for (x, y, w, h) in faces], dtype=np.int32)

    # 绘制人脸框
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 检测关键点（仅当检测到人脸时）
    if len(faces_np) > 0:
        ok, landmarks = facemark.fit(gray, faces=faces_np)
        if ok:
            for landmark in landmarks:
                for (x, y) in landmark[0].astype(int):  # 确保坐标是整数
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow('Face Landmarks', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
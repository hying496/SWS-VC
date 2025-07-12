import cv2
import numpy as np

# 加载 Haar 级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 初始化 LBF 面部关键点检测器
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")  # 确保路径正确

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 原始灰度图（未处理）
    gray_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 预处理：伽马校正 + CLAHE
    gray_proc = adjust_gamma(gray_orig, gamma=1.5)
    gray_proc = clahe.apply(gray_proc)

    # 复制原始帧，分别在两个窗口上绘制结果
    frame_orig = frame.copy()
    frame_proc = frame.copy()

    # 未处理版本检测
    faces_orig = face_cascade.detectMultiScale(gray_orig, scaleFactor=1.1, minNeighbors=5)
    faces_np_orig = np.array([[x, y, w, h] for (x, y, w, h) in faces_orig], dtype=np.int32)
    for (x, y, w, h) in faces_orig:
        cv2.rectangle(frame_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if len(faces_np_orig) > 0:
        ok, landmarks = facemark.fit(gray_orig, faces=faces_np_orig)
        if ok:
            for landmark in landmarks:
                for (x, y) in landmark[0].astype(int):
                    cv2.circle(frame_orig, (x, y), 2, (0, 0, 255), -1)

    # 处理后版本检测
    faces_proc = face_cascade.detectMultiScale(gray_proc, scaleFactor=1.1, minNeighbors=5)
    faces_np_proc = np.array([[x, y, w, h] for (x, y, w, h) in faces_proc], dtype=np.int32)
    for (x, y, w, h) in faces_proc:
        cv2.rectangle(frame_proc, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if len(faces_np_proc) > 0:
        ok, landmarks = facemark.fit(gray_proc, faces=faces_np_proc)
        if ok:
            for landmark in landmarks:
                for (x, y) in landmark[0].astype(int):
                    cv2.circle(frame_proc, (x, y), 2, (0, 0, 255), -1)

    # 拼接两个结果画面（宽度方向并排）
    combined_frame = np.hstack((frame_orig, frame_proc))

    # 在画面上加文字标注
    cv2.putText(combined_frame, 'Original', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(combined_frame, 'Gamma+CLAHE', (frame_orig.shape[1] + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Comparison: Original vs Gamma+CLAHE', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

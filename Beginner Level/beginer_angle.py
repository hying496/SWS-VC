import cv2
import dlib
import numpy as np
import mediapipe as mp
import time

# ----------- 初始化部分 -----------

# Haar + LBF
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")

# Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# 摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 摄像头打开失败")
    exit()

# ----------- 主循环 -----------
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ 读取帧失败，跳过")
        continue

    # 复制三份图像
    frame_lbf = frame.copy()
    frame_dlib = frame.copy()
    frame_mp = frame.copy()

    gray_lbf = cv2.cvtColor(frame_lbf, cv2.COLOR_BGR2GRAY)
    gray_dlib = cv2.cvtColor(frame_dlib, cv2.COLOR_BGR2GRAY)
    rgb_mp = cv2.cvtColor(frame_mp, cv2.COLOR_BGR2RGB)

    # ----------- Haar + LBF -----------
    try:
        faces = face_cascade.detectMultiScale(gray_lbf, 1.1, 5)
        faces_np = np.array([[x, y, w, h] for (x, y, w, h) in faces], dtype=np.int32)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_lbf, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if len(faces_np) > 0:
            ok, landmarks = facemark.fit(gray_lbf, faces=faces_np)
            if ok:
                for landmark in landmarks:
                    for (x, y) in landmark[0].astype(int):
                        cv2.circle(frame_lbf, (x, y), 1, (0, 0, 255), -1)
    except:
        pass

    cv2.putText(frame_lbf, "Haar + LBF", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # ----------- Dlib -----------
    try:
        faces_dlib = detector(gray_dlib)
        for face in faces_dlib:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(frame_dlib, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(gray_dlib, face)
            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame_dlib, (x, y), 1, (0, 0, 255), -1)
    except:
        pass

    cv2.putText(frame_dlib, "Dlib 68", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # ----------- MediaPipe FaceMesh -----------
    try:
        results = face_mesh.process(rgb_mp)
        if results.multi_face_landmarks:
            h, w, _ = frame_mp.shape
            for face_landmarks in results.multi_face_landmarks:
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame_mp, (x, y), 1, (0, 255, 255), -1)
    except:
        print("⚠️ MediaPipe 处理失败，跳过帧")
        pass

    cv2.putText(frame_mp, "MediaPipe", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # ----------- 合并三帧并显示 -----------
    height = 360
    frame_lbf = cv2.resize(frame_lbf, (480, height))
    frame_dlib = cv2.resize(frame_dlib, (480, height))
    frame_mp = cv2.resize(frame_mp, (480, height))

    combined = np.hstack((frame_lbf, frame_dlib, frame_mp))

    # 显示 FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time
    cv2.putText(combined, f"FPS: {int(fps)}", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Landmark Comparison", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------- 清理 -----------

cap.release()
cv2.destroyAllWindows()

import cv2
import dlib
import numpy as np
import joblib

# ------------------------- 配置路径 -------------------------
predictor_path = "shape_predictor_68_face_landmarks.dat"
model_path = "D:\\face_expression_self\\svm_emotion_model.pkl"  # 替换为你的模型路径

# ------------------------- 加载模型 -------------------------
print("[INFO] Loading model …")
clf = joblib.load(model_path)
expected_dim = clf.n_features_in_
print(f"[INFO] Model expects feature dim: {expected_dim}")

# ------------------------- 初始化 dlib -------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# ------------------------- 特征处理函数 -------------------------
def extract_normalized_landmark_features(shape: dlib.full_object_detection) -> np.ndarray:
    """将 dlib 的68个关键点中心化、归一化，返回 136维特征"""
    coords = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)

    # 中心化
    center = np.mean(coords, axis=0)
    coords -= center

    # 归一化
    norm = np.linalg.norm(coords)
    if norm > 0:
        coords /= norm

    return coords.flatten()  # 展平为 (136,)

# ------------------------- 实时检测 -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("[INFO] Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        feats = extract_normalized_landmark_features(shape)

        if feats.shape[0] != expected_dim:
            print(f"[WARN] Feature dim mismatch: got {feats.shape[0]}, expected {expected_dim}")
            continue

        # 预测
        proba = clf.predict_proba([feats])[0]
        idx = np.argmax(proba)
        label = clf.classes_[idx]
        confidence = proba[idx]

        # 绘制结果
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label}: {confidence:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Real-time Expression Detection (dlib)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

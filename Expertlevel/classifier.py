import argparse
import glob
import os
from pathlib import Path
import cv2
import dlib
import joblib
import numpy as np
import seaborn as sns
import mediapipe as mp
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --------------- Configurable paths and constants --------------- #
CASCADE_PATH = "haarcascade_frontalface_default.xml"
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# If you changed the emotion folder names, edit this list accordingly
EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]
def plot_confusion_matrix(y_true, y_pred, labels, normalize=True, cmap="Blues"):
    cm =confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm,
                annot=True,
                fmt=".2f" if normalize else "d",
                cmap=cmap,
                xticklabels=labels,
                yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    plt.tight_layout()
    plt.show()
def extract_features_from_mediapipe_478_2(face_landmarks, image_width, image_height) -> np.ndarray:
    """
    将 mediapipe 检测到的 478 个关键点转为 956 维向量（单位：像素坐标）
    """
    coords = []
    for i, lm in enumerate(face_landmarks.landmark):
        if i >= 478:
            break
        coords.extend([lm.x * image_width, lm.y * image_height])
    return np.array(coords, dtype=np.float32)
# ------------------------- Data loading ------------------------- #
def load_dataset(landmarks_dir: str):
    """
    加载 landmarks 数据集，假设 landmarks_dir 下按类别分子文件夹存放 .npy 文件。
    """
    X, y = [], []
    landmarks_path = Path(landmarks_dir)

    npy_files = list(landmarks_path.rglob("*.npy"))
    if not npy_files:
        raise RuntimeError(f"No landmark files found under {landmarks_dir}")

    for file in npy_files:
        try:
            landmarks = np.load(file)
            # 扁平化为一维数组作为 SVM 特征
            X.append(landmarks.flatten())
            # 类别标签 = 上一级文件夹名
            y.append(file.parent.name)
        except Exception as e:
            print(f"[WARN] Failed to load {file}: {e}")
            continue

    return np.array(X), np.array(y)
def train_svm(landmarks_dir: str, model_path: str):
    print("[INFO] Loading dataset…")
    X, y = load_dataset(landmarks_dir)
    print(f"[INFO] Loaded {len(X)} samples across {len(set(y))} classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    print("[INFO] Training SVM (RBF kernel)…")
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    )
    clf.fit(X_train, y_train)

    print("[INFO] Evaluating…")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))

    print("[INFO] Plotting confusion matrix…")
    plot_confusion_matrix(y_test, y_pred, labels=np.unique(y))

    joblib.dump(clf, model_path)
    print(f"[INFO] Saved model to {model_path}")
# ------------------------- Main entry ----------------------
class Args:
    mode = "train"
    landmarks_dir = "D:\\face_expression_self\\output_file_CN_dlib"
    model_path = "D:/face_expression_self/svm_emotion_model.pkl"
if __name__ == "__main__":
   args = Args()
   train_svm(args.landmarks_dir, args.model_path)





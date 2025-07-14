#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch face detection & 68‑point landmark annotation using dlib only.

For every image under DATASET_ROOT:
1. detect the largest face (HOG or CNN detector),
2. predict 68 landmark points,
3. optionally crop/annotate and save a PNG,
4. save landmarks as a NumPy array (.npy) that mirrors the dataset folder tree.

Edit the CONFIG section before running.
"""

import os
from pathlib import Path
import cv2
import dlib
import numpy as np

# --------------------------- CONFIG --------------------------- #
DATASET_ROOT   = Path(r"D:\face_expression_self\fer2013_sr_192x192_train")  # 输入数据根目录

# 输出目录
OUT_IMAGE_DIR      = Path("annotated_fullface_dlib")   # 带关键点 PNG；None 表示不保存
OUT_LANDMARK_DIR   = Path("output_file_dlib")         # 保存 .npy

# 模型
PREDICTOR_PATH     = Path("shape_predictor_68_face_landmarks.dat")    # 68 点
USE_CNN_DETECTOR   = False                                            # True → 用 CNN
CNN_MODEL_PATH     = Path("mmod_human_face_detector.dat")             # 若 USE_CNN_DETECTOR = True
# -------------------------------------------------------------- #


# ---------------------- 模型加载 ------------------------------- #
print("[INFO] Loading models…")
predictor = dlib.shape_predictor(str(PREDICTOR_PATH))

if USE_CNN_DETECTOR:
    cnn_detector = dlib.cnn_face_detection_model_v1(str(CNN_MODEL_PATH))
else:
    hog_detector = dlib.get_frontal_face_detector()
# -------------------------------------------------------------- #


# -------------------- 辅助函数 --------------------------- #
def detect_largest_face(img_bgr):
    """Return largest dlib.rectangle or None."""
    if USE_CNN_DETECTOR:
        # CNN 返回 mmod_rectangles；取 .rect
        dets = cnn_detector(img_bgr, 1)
        if not dets:
            return None
        rect = max(dets, key=lambda r: r.rect.width() * r.rect.height()).rect
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        dets = hog_detector(gray, 1)
        if not dets:
            return None
        rect = max(dets, key=lambda r: r.width() * r.height())
    return rect


def get_landmarks(img_bgr, rect):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    shape = predictor(img_rgb, rect)
    return np.array([[pt.x, pt.y] for pt in shape.parts()], dtype=np.int32)


def draw_landmarks(img_bgr, landmarks, draw_idx=False):
    canvas = img_bgr.copy()
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(canvas, (x, y), 2, (0, 255, 255), -1)
        if draw_idx:
            cv2.putText(canvas, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 255), 1, cv2.LINE_AA)
    return canvas
# --------------------------------------------------------- #


def process_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError("Cannot read image")

    rect = detect_largest_face(img)
    if rect is None:
        raise RuntimeError("No face detected")

    lmks = get_landmarks(img, rect)

    # 视图裁剪（可选）
    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
    crop = img[max(y1, 0):y2, max(x1, 0):x2]
    annotated = draw_landmarks(crop, lmks - np.array([x1, y1]))

    return annotated, lmks


def process_dataset(root_dir, out_img_dir, out_lmk_dir):
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    total = success = failure = 0

    for path in Path(root_dir).rglob("*"):
        if path.suffix.lower() not in img_exts:
            continue
        total += 1
        try:
            annotated, lmks = process_image(path)

            # 还原相对路径
            rel = path.relative_to(root_dir)
            rel_dir = rel.parent

            # ---- 保存 landmarks (.npy) ----
            lmk_path = (out_lmk_dir / rel_dir / (path.stem + ".npy"))
            lmk_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(lmk_path, lmks)

            # ---- 保存可视化 PNG（可选）----
            if out_img_dir:
                img_path = (out_img_dir / rel_dir / (path.stem + "_dlib.png"))
                img_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(img_path), annotated)

            success += 1
            print(f"[✓] {rel}")
        except Exception as e:
            print(f"[×] {path}: {e}")
            failure += 1

    print(f"\n[INFO] Done. {success}/{total} succeeded, {failure} failed.")


if __name__ == "__main__":
    if OUT_IMAGE_DIR:
        OUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LANDMARK_DIR.mkdir(parents=True, exist_ok=True)

    process_dataset(DATASET_ROOT, OUT_IMAGE_DIR, OUT_LANDMARK_DIR)
"""Full‑face landmark detection over a dataset using Haar + dlib

For every image in `DATASET_ROOT` the script will
1. locate the largest face with OpenCV’s Haar cascade,
2. predict 68 landmark points with dlib,
3. (optionally) crop the detected face region,
4. draw the landmarks on that crop,
5. save the annotated crop alongside a NumPy file containing the raw landmark coordinates.

Change the paths below to point at your own dataset and model files.
"""

import os
from pathlib import Path
import cv2
import dlib
import numpy as np

# --------------------- Configuration --------------------- #
CASCADE_PATH = "haarcascade_frontalface_default.xml"  # Haar model
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  # dlib 68‑pt model

# Root of dataset (update to your own path)
DATASET_ROOT = r"D:\face_expression_self\fer2013_sr_192x192"

# Where to save outputs
OUT_IMAGE_DIR = "annotated_fullface"      # annotated PNGs
OUT_LANDMARK_DIR = "landmark_npy1"          # 68×2 NumPy arrays

# Create output folders if they do not exist
Path(OUT_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
Path(OUT_LANDMARK_DIR).mkdir(parents=True, exist_ok=True)

# --------------------- Load models ----------------------- #
cascade = cv2.CascadeClassifier(CASCADE_PATH)
if cascade.empty():
    raise IOError(f"Cannot load Haar cascade from {CASCADE_PATH}")

predictor = dlib.shape_predictor(PREDICTOR_PATH)

# --------------------- Helper functions ------------------ #

def detect_largest_face(gray: np.ndarray):
    """Return the largest detected face as a dlib.rectangle or None."""
    rects = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(rects) == 0:
        return None
    x, y, w, h = max(rects, key=lambda r: r[2] * r[3])
    return dlib.rectangle(int(x), int(y), int(x + w), int(y + h))


def get_landmarks(img_bgr: np.ndarray, rect: dlib.rectangle):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    shape = predictor(img_rgb, rect)
    return np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32)


def draw_landmarks(img_bgr: np.ndarray, landmarks: np.ndarray, draw_idx: bool = False):
    out = img_bgr.copy()
    for idx, (x, y) in enumerate(landmarks):
        cv2.circle(out, (x, y), radius=2, thickness=-1, color=(0, 255, 255))
        if draw_idx:
            cv2.putText(out, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def process_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rect = detect_largest_face(gray)
    if rect is None:
        raise RuntimeError("No face detected")

    landmarks = get_landmarks(img, rect)

    # Crop face ROI for visualization (keeps file sizes small)
    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
    face_roi = img[max(y1, 0):y2, max(x1, 0):x2]

    annotated = draw_landmarks(face_roi, landmarks - np.array([x1, y1]))
    return annotated, landmarks


def process_dataset(root_dir: str):
    total = success = failure = 0
    for root, _, files in os.walk(root_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                continue
            total += 1
            fpath = os.path.join(root, fname)
            try:
                annotated, landmarks = process_image(fpath)

                # Re‑create relative folder structure
                rel_path = os.path.relpath(fpath, root_dir)
                rel_dir = os.path.dirname(rel_path)

                # ---- Save annotated image ---- #
                out_img_path = Path(OUT_IMAGE_DIR, rel_dir, Path(fname).stem + '_fullface.png')
                out_img_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_img_path), annotated)

                # ---- Save landmarks (68×2) ---- #
                out_lmk_path = Path(OUT_LANDMARK_DIR, rel_dir, Path(fname).stem + '.npy')
                out_lmk_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_lmk_path, landmarks)

                success += 1
            except Exception as e:
                print(f"[ERROR] {fpath}: {e}")
                failure += 1
    print(f"Done. {success}/{total} images processed successfully, {failure} failed.")


if __name__ == '__main__':
    process_dataset(DATASET_ROOT)

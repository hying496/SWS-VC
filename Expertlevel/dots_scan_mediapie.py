import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp
#C:\Users\HWPC\AppData\Local\Programs\Python\Python311\python.exe

# -------------------- 参数配置 -------------------- #
INPUT_DIR = "D:\\face_expression_self\\fer2013_sr_192x192_test"  # 输入图像目录
OUTPUT_DIR = "/landmarks_mp_test"  # 输出目录
SAVE_TXT = True         # 是否保存 .txt 文件
SAVE_VIS = True         # 是否保存可视化图像

# -------------------- 初始化 MediaPipe -------------------- #
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5)

# -------------------- 创建输出目录 -------------------- #
Path(OUTPUT_DIR, "npy").mkdir(parents=True, exist_ok=True)
if SAVE_TXT:
    Path(OUTPUT_DIR, "txt").mkdir(parents=True, exist_ok=True)
if SAVE_VIS:
    Path(OUTPUT_DIR, "vis").mkdir(parents=True, exist_ok=True)

# -------------------- 图像处理 -------------------- #
image_paths = list(Path(INPUT_DIR).rglob("*.jpg")) + list(Path(INPUT_DIR).rglob("*.png"))

for img_path in tqdm(image_paths, desc="Processing"):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Cannot read {img_path}")
            continue
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            continue  # 跳过无人脸图像

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark],
                             dtype=np.float32)

        # -------- 1) 生成与 INPUT_DIR 同构的子路径 -------- #
        rel_path = img_path.relative_to(INPUT_DIR)         # e.g. angry/0001.jpg
        rel_dir  = rel_path.parent                         # e.g. angry

        # -------- 2) 针对三种输出类型分别创建子目录 -------- #
        npy_dir  = Path(OUTPUT_DIR, "npy", rel_dir)
        txt_dir  = Path(OUTPUT_DIR, "txt", rel_dir)
        vis_dir  = Path(OUTPUT_DIR, "vis", rel_dir)

        npy_dir.mkdir(parents=True, exist_ok=True)
        if SAVE_TXT:
            txt_dir.mkdir(parents=True, exist_ok=True)
        if SAVE_VIS:
            vis_dir.mkdir(parents=True, exist_ok=True)

        # -------- 3) 保存文件 -------- #
        fname = img_path.stem                               # 不带扩展名
        np.save(npy_dir / f"{fname}.npy", landmarks)

        if SAVE_TXT:
            np.savetxt(txt_dir / f"{fname}.txt", landmarks, fmt="%.2f")

        if SAVE_VIS:
            vis_img = img.copy()
            for x, y in landmarks:
                cv2.circle(vis_img, (int(x), int(y)), 1, (0, 255, 0), -1)
            cv2.imwrite(vis_dir / f"{fname}.jpg", vis_img)

    except Exception as e:
        print(f"[ERROR] Failed to process {img_path}: {e}")

print("✅ 完成！已按原始类别结构输出 landmarks。")

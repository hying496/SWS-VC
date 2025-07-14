import numpy as np
from pathlib import Path


# ====================== 参数配置 ====================== #
INPUT_DIR = Path("D:\\face_expression_self\\output_file_dlib")         # 输入目录
OUTPUT_DIR = Path("D:\\face_expression_self\\output_file_CN_dlib")   # 输出目录
CENTER_INDEX = 1      # MediaPipe 鼻尖索引
NORM_METHOD = "max"   # 可选："max", "bbox", "std"
# ===================================================== #


def normalize_centered(points: np.ndarray, method: str = "max") -> np.ndarray:
    if method == "max":
        scale = np.max(np.linalg.norm(points, axis=1))
    elif method == "bbox":
        scale = np.max(np.abs(points))
    elif method == "std":
        scale = np.std(points)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return points / (scale + 1e-8)


def process_landmark_file(landmark_array: np.ndarray, center_idx: int, norm: str) -> np.ndarray:
    if landmark_array.ndim == 1:
        landmark_array = landmark_array.reshape(-1, 2)
    center_point = landmark_array[center_idx]
    centered = landmark_array - center_point
    normalized = normalize_centered(centered, method=norm)
    return normalized


def batch_process_recursive(input_dir: Path, output_dir: Path, center_idx: int, norm: str):
    npy_files = list(input_dir.rglob("*.npy"))
    if not npy_files:
        print(f"[警告] 没有找到 .npy 文件于目录: {input_dir}")
        return

    print(f"[INFO] 发现 {len(npy_files)} 个 .npy 文件，开始处理...")

    for file in npy_files:
        relative_path = file.relative_to(input_dir)
        output_file = output_dir / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = np.load(file)
            normed = process_landmark_file(data, center_idx, norm)
            np.save(output_file, normed)
            print(f"[✓] {relative_path}")
        except Exception as e:
            print(f"[ERROR] 处理 {file} 时失败：{e}")

    print(f"\n[完成] 全部处理完毕，结果已保存到: {output_dir.resolve()}")


if __name__ == "__main__":
    batch_process_recursive(INPUT_DIR, OUTPUT_DIR, CENTER_INDEX, NORM_METHOD)


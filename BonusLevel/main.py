# Import necessary libraries
import sys
import os
import cv2
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import mediapipe as mp

# Suppress MediaPipe warnings
import logging

logging.getLogger('mediapipe').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MediaPipePoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MediaPipe Dance GUI")
        self.root.geometry("1500x800")  # 调整窗口大小以适应Canvas布局
        self.root.configure(bg='lightgray')

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.running_file = False
        self.running_cam = False
        self.video_path = ""
        self.cap_file = None
        self.cap_cam = None
        self.show_video_frame = True

        self.setup_gui()

    def setup_gui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='lightgray')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame for reference video
        self.left_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, bd=2)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Right frame for webcam
        self.right_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, bd=2)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Left side - Reference Video
        tk.Label(self.left_frame, text="Reference Video", font=("Arial", 16, "bold"), bg='white').pack(pady=5)

        # 创建视频显示框架，限制高度
        video_frame = tk.Frame(self.left_frame, bg='white')
        video_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 使用Canvas来精确控制显示尺寸
        self.canvas_file = tk.Canvas(video_frame, width=700, height=525, bg="black")
        self.canvas_file.pack(pady=5)
        self.canvas_file.create_text(350, 262, text="No video loaded", fill="white", font=("Arial", 12))

        # Left side controls - 确保按钮始终可见
        self.controls_file = tk.Frame(self.left_frame, bg='white')
        self.controls_file.pack(side=tk.BOTTOM, pady=10)

        tk.Button(self.controls_file, text="📁 Open Video", command=self.load_video,
                  bg='lightblue', font=("Arial", 10), width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(self.controls_file, text="▶️ Start Video", command=self.start_video,
                  bg='lightgreen', font=("Arial", 10), width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(self.controls_file, text="⏹️ Stop Video", command=self.stop_video,
                  bg='lightcoral', font=("Arial", 10), width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(self.controls_file, text="🔄 Show/Hide", command=self.toggle_video_display,
                  bg='lightyellow', font=("Arial", 10), width=12).pack(side=tk.LEFT, padx=2)

        # Right side - Webcam
        tk.Label(self.right_frame, text="Your Webcam", font=("Arial", 16, "bold"), bg='white').pack(pady=5)

        # 创建摄像头显示框架，限制高度
        cam_frame = tk.Frame(self.right_frame, bg='white')
        cam_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 使用Canvas来精确控制显示尺寸
        self.canvas_cam = tk.Canvas(cam_frame, width=700, height=525, bg="black")
        self.canvas_cam.pack(pady=5)
        self.canvas_cam.create_text(350, 262, text="Webcam not started", fill="white", font=("Arial", 12))

        # Right side controls - 确保按钮始终可见
        self.controls_cam = tk.Frame(self.right_frame, bg='white')
        self.controls_cam.pack(side=tk.BOTTOM, pady=10)

        tk.Button(self.controls_cam, text="📷 Start Webcam", command=self.start_cam,
                  bg='lightgreen', font=("Arial", 10), width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(self.controls_cam, text="⏹️ Stop Webcam", command=self.stop_cam,
                  bg='lightcoral', font=("Arial", 10), width=15).pack(side=tk.LEFT, padx=5)

        # Bottom info frame
        self.info_frame = tk.Frame(self.root, bg='lightgray')
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        self.info_label = tk.Label(self.info_frame,
                                   text="✅ Ready to start! Click 'Start Webcam' to test pose detection.",
                                   font=("Arial", 12), bg='lightgray', fg='darkgreen')
        self.info_label.pack()

        # Status indicators
        self.status_frame = tk.Frame(self.info_frame, bg='lightgray')
        self.status_frame.pack(pady=5)

        self.video_status = tk.Label(self.status_frame, text="📹 Video: Stopped",
                                     font=("Arial", 10), bg='lightgray')
        self.video_status.pack(side=tk.LEFT, padx=10)

        self.cam_status = tk.Label(self.status_frame, text="📷 Webcam: Stopped",
                                   font=("Arial", 10), bg='lightgray')
        self.cam_status.pack(side=tk.LEFT, padx=10)

    def load_video(self):
        path = filedialog.askopenfilename(
            title="Select Dance Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            self.video_path = path
            filename = os.path.basename(path)
            self.info_label.config(text=f"✅ Video loaded: {filename}")
            # 在Canvas上显示文本
            self.canvas_file.delete("all")
            self.canvas_file.create_text(350, 262, text=f"Video loaded:\n{filename}",
                                         fill="white", font=("Arial", 12))

    def start_video(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please select a video file first!")
            return
        if not self.running_file:
            self.running_file = True
            self.video_status.config(text="📹 Video: Playing", fg='green')
            self.info_label.config(text="🎬 Video playing! You can now compare with webcam.")
            threading.Thread(target=self.process_video_file, daemon=True).start()

    def stop_video(self):
        self.running_file = False
        self.video_status.config(text="📹 Video: Stopped", fg='red')
        self.info_label.config(text="⏹️ Video stopped.")
        # 在Canvas上显示停止文本
        self.canvas_file.delete("all")
        self.canvas_file.create_text(350, 262, text="Video stopped", fill="white", font=("Arial", 12))
        if self.cap_file:
            self.cap_file.release()

    def toggle_video_display(self):
        self.show_video_frame = not self.show_video_frame
        mode = "Original video" if self.show_video_frame else "Pose skeleton only"
        self.info_label.config(text=f"🔄 Display mode: {mode}")

    def start_cam(self):
        if not self.running_cam:
            self.running_cam = True
            self.cam_status.config(text="📷 Webcam: Running", fg='green')
            self.info_label.config(text="📷 Webcam started! Move around to test pose detection.")
            threading.Thread(target=self.process_webcam, daemon=True).start()

    def stop_cam(self):
        self.running_cam = False
        self.cam_status.config(text="📷 Webcam: Stopped", fg='red')
        self.info_label.config(text="📷 Webcam stopped.")
        # 在Canvas上显示停止文本
        self.canvas_cam.delete("all")
        self.canvas_cam.create_text(350, 262, text="Webcam stopped", fill="white", font=("Arial", 12))
        if self.cap_cam:
            self.cap_cam.release()

    def process_video_file(self):
        try:
            self.cap_file = cv2.VideoCapture(self.video_path)
            if not self.cap_file.isOpened():
                messagebox.showerror("Error", "Cannot open video file!")
                return

            fps = self.cap_file.get(cv2.CAP_PROP_FPS) or 30
            delay = max(1, int(1000 / fps))

            while self.cap_file.isOpened() and self.running_file:
                ret, frame = self.cap_file.read()
                if not ret:
                    # Video ended, restart from beginning
                    self.cap_file.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                processed_frame = self.process_pose(frame)
                self.update_canvas(self.canvas_file, processed_frame)

                cv2.waitKey(delay)

        except Exception as e:
            messagebox.showerror("Video Error", f"Error processing video: {str(e)}")
        finally:
            if self.cap_file:
                self.cap_file.release()

    def process_webcam(self):
        try:
            self.cap_cam = cv2.VideoCapture(0)
            if not self.cap_cam.isOpened():
                messagebox.showerror("Error", "Cannot open webcam!")
                return

            while self.cap_cam.isOpened() and self.running_cam:
                ret, frame = self.cap_cam.read()
                if not ret:
                    continue

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                processed_frame = self.process_pose(frame)
                self.update_canvas(self.canvas_cam, processed_frame)

        except Exception as e:
            messagebox.showerror("Webcam Error", f"Error processing webcam: {str(e)}")
        finally:
            if self.cap_cam:
                self.cap_cam.release()

    def process_pose(self, frame):
        """Process frame with MediaPipe pose detection"""
        height, width = frame.shape[:2]

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.pose.process(rgb_frame)

        # Create output frame
        if self.show_video_frame:
            output_frame = frame.copy()
        else:
            output_frame = np.ones_like(frame) * 255  # White background

        # Draw pose landmarks
        if results.pose_landmarks:
            # Draw landmarks and connections
            self.mp_drawing.draw_landmarks(
                output_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

            # Add pose detected indicator
            cv2.putText(output_frame, "POSE DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # No pose detected
            cv2.putText(output_frame, "NO POSE DETECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return output_frame

    def update_canvas(self, canvas, frame):
        """Update tkinter canvas with processed frame"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((700, 525))  # 固定尺寸匹配Canvas
            imgtk = ImageTk.PhotoImage(image=img)

            # 清除之前的内容
            canvas.delete("all")
            # 在Canvas中心显示图像
            canvas.create_image(350, 262, image=imgtk)
            # 保持引用防止垃圾回收
            canvas.imgtk = imgtk
        except Exception as e:
            print(f"Error updating canvas: {e}")

    def cleanup(self):
        """Cleanup resources"""
        self.running_file = False
        self.running_cam = False
        if self.cap_file:
            self.cap_file.release()
        if self.cap_cam:
            self.cap_cam.release()
        if hasattr(self, 'pose'):
            self.pose.close()


def main():
    # Check MediaPipe installation
    try:
        import mediapipe as mp
        print("✅ MediaPipe is available!")
    except ImportError:
        print("❌ MediaPipe is not installed. Please install it using: pip install mediapipe")
        return

    # Create and run app
    root = tk.Tk()
    app = MediaPipePoseApp(root)

    # Handle window close event
    def on_closing():
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    print("🚀 Starting MediaPipe Dance GUI...")
    print("💡 Tip: Click 'Start Webcam' first to test pose detection!")

    root.mainloop()


if __name__ == "__main__":
    main()
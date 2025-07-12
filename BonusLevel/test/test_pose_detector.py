# test_pose_detector.py
# 完整测试pose_detector模块

import cv2
import numpy as np
import sys
import os
import time
from pathlib import Path

# 添加项目路径 - 从test文件夹导入上级目录的模块
project_root = Path(__file__).parent.parent  # Bonus Level 目录
sys.path.insert(0, str(project_root))
print(f"🔍 项目根目录: {project_root}")

try:
    from detector.pose_detector import (
        PoseDetectionManager,
        DetectorType,
        DetectorFactory
    )

    print("✅ 成功导入pose_detector模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保：")
    print("1. detector/pose_detector.py 文件存在")
    print("2. detector/__init__.py 文件存在（可以是空文件）")
    sys.exit(1)


class PoseDetectorTester:
    """姿态检测器测试类"""

    def __init__(self):
        self.test_results = {}
        print("🧪 姿态检测器测试器初始化")

    def test_1_basic_import(self):
        """测试1: 基础导入测试"""
        print("\n" + "=" * 50)
        print("🧪 测试1: 基础导入测试")

        try:
            # 测试所有检测器类型
            available_detectors = DetectorFactory.get_available_detectors()
            print(f"✅ 可用检测器: {[d.value for d in available_detectors]}")

            self.test_results['import'] = True
            return True
        except Exception as e:
            print(f"❌ 导入测试失败: {e}")
            self.test_results['import'] = False
            return False

    def test_2_mediapipe_detector(self):
        """测试2: MediaPipe检测器"""
        print("\n" + "=" * 50)
        print("🧪 测试2: MediaPipe检测器")

        try:
            # 创建MediaPipe检测器
            manager = PoseDetectionManager(DetectorType.MEDIAPIPE)
            print("✅ MediaPipe检测器创建成功")

            # 获取模型信息
            info = manager.detector.get_model_info()
            print(f"📋 模型信息: {info}")

            # 测试空帧检测
            empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            persons, detection_info = manager.detect_poses(empty_frame)

            print(f"📊 空帧检测结果:")
            print(f"   检测到人数: {detection_info['persons_count']}")
            print(f"   处理时间: {detection_info['processing_time_ms']:.2f}ms")

            manager.cleanup()
            self.test_results['mediapipe'] = True
            return True

        except Exception as e:
            print(f"❌ MediaPipe测试失败: {e}")
            self.test_results['mediapipe'] = False
            return False

    def test_3_webcam_detection(self):
        """测试3: 摄像头实时检测"""
        print("\n" + "=" * 50)
        print("🧪 测试3: 摄像头实时检测")
        print("按 'q' 退出，按 's' 截图保存")

        try:
            # 创建检测器
            manager = PoseDetectionManager(DetectorType.MEDIAPIPE)

            # 打开摄像头
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ 无法打开摄像头")
                self.test_results['webcam'] = False
                return False

            print("✅ 摄像头已打开")

            frame_count = 0
            fps_history = []
            screenshot_count = 0

            while True:
                start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    print("❌ 无法读取摄像头画面")
                    break

                # 姿态检测
                persons, detection_info = manager.detect_poses(frame)

                # 计算FPS
                fps = 1.0 / (time.time() - start_time)
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)

                # 绘制检测结果
                vis_frame = self.draw_detection_results(frame, persons, detection_info, fps_history)

                # 显示
                cv2.imshow('Pose Detection Test', vis_frame)

                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_count += 1
                    filename = f"test_screenshot_{screenshot_count}.jpg"
                    cv2.imwrite(filename, vis_frame)
                    print(f"📸 截图保存: {filename}")

                frame_count += 1

                # 每30帧打印一次统计
                if frame_count % 30 == 0:
                    avg_fps = np.mean(fps_history) if fps_history else 0
                    stats = manager.get_performance_stats()
                    print(f"📊 第{frame_count}帧: FPS={avg_fps:.1f}, "
                          f"成功率={stats['success_rate']:.1f}%, "
                          f"检测到{detection_info['persons_count']}人")

            cap.release()
            cv2.destroyAllWindows()
            manager.cleanup()

            print(f"✅ 摄像头测试完成，总共处理{frame_count}帧")
            self.test_results['webcam'] = True
            return True

        except Exception as e:
            print(f"❌ 摄像头测试失败: {e}")
            self.test_results['webcam'] = False
            return False

    def test_4_performance_benchmark(self):
        """测试4: 性能基准测试"""
        print("\n" + "=" * 50)
        print("🧪 测试4: 性能基准测试")

        try:
            manager = PoseDetectionManager(DetectorType.MEDIAPIPE)

            # 创建不同尺寸的测试帧
            test_frames = {
                "小尺寸 (320x240)": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
                "中尺寸 (640x480)": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                "大尺寸 (1280x720)": np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
            }

            performance_results = {}

            for size_name, test_frame in test_frames.items():
                print(f"🔍 测试 {size_name}...")

                processing_times = []

                # 运行10次测试
                for i in range(10):
                    start_time = time.time()
                    persons, detection_info = manager.detect_poses(test_frame)
                    processing_time = (time.time() - start_time) * 1000
                    processing_times.append(processing_time)

                avg_time = np.mean(processing_times)
                fps_estimate = 1000 / avg_time

                performance_results[size_name] = {
                    'avg_time_ms': avg_time,
                    'fps_estimate': fps_estimate,
                    'min_time_ms': min(processing_times),
                    'max_time_ms': max(processing_times)
                }

                print(f"   平均处理时间: {avg_time:.2f}ms")
                print(f"   预估FPS: {fps_estimate:.1f}")
                print(f"   时间范围: {min(processing_times):.2f}ms - {max(processing_times):.2f}ms")

            manager.cleanup()

            # 保存性能报告
            self.save_performance_report(performance_results)

            self.test_results['performance'] = True
            return True

        except Exception as e:
            print(f"❌ 性能测试失败: {e}")
            self.test_results['performance'] = False
            return False

    def test_5_detector_switching(self):
        """测试5: 检测器切换测试"""
        print("\n" + "=" * 50)
        print("🧪 测试5: 检测器切换测试")

        try:
            # 获取可用检测器
            available_detectors = DetectorFactory.get_available_detectors()
            print(f"可用检测器: {[d.value for d in available_detectors]}")

            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            for detector_type in available_detectors:
                print(f"🔄 测试 {detector_type.value} 检测器...")

                try:
                    manager = PoseDetectionManager(detector_type)
                    persons, detection_info = manager.detect_poses(test_frame)

                    print(f"   ✅ {detector_type.value}: "
                          f"检测到{detection_info['persons_count']}人, "
                          f"用时{detection_info['processing_time_ms']:.2f}ms")

                    manager.cleanup()

                except Exception as e:
                    print(f"   ❌ {detector_type.value}失败: {e}")

            self.test_results['switching'] = True
            return True

        except Exception as e:
            print(f"❌ 检测器切换测试失败: {e}")
            self.test_results['switching'] = False
            return False

    def draw_detection_results(self, frame, persons, detection_info, fps_history):
        """绘制检测结果"""
        vis_frame = frame.copy()
        height, width = frame.shape[:2]

        # 绘制姿态
        for person in persons:
            self.draw_simple_pose(vis_frame, person)

        # 绘制信息面板
        self.draw_info_panel(vis_frame, detection_info, fps_history)

        return vis_frame

    def draw_simple_pose(self, frame, person):
        """绘制简单的姿态"""
        keypoints = person.keypoints

        # MediaPipe连接关系（简化）
        connections = [
            (5, 6),  # 肩膀
            (5, 7), (7, 9),  # 左臂
            (6, 8), (8, 10),  # 右臂
            (5, 11), (6, 12), (11, 12),  # 躯干
            (11, 13), (13, 15),  # 左腿
            (12, 14), (14, 16),  # 右腿
        ]

        # 绘制连接线
        for start_idx, end_idx in connections:
            if (start_idx < len(keypoints) and end_idx < len(keypoints)):
                start_kp = keypoints[start_idx]
                end_kp = keypoints[end_idx]

                if start_kp.visible and end_kp.visible:
                    start_point = (int(start_kp.x), int(start_kp.y))
                    end_point = (int(end_kp.x), int(end_kp.y))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

        # 绘制关键点
        for kp in keypoints:
            if kp.visible:
                color = (0, 255, 0) if kp.confidence > 0.5 else (0, 255, 255)
                cv2.circle(frame, (int(kp.x), int(kp.y)), 4, color, -1)

        # 绘制边界框
        x1, y1, x2, y2 = person.bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # 人员ID和置信度
        cv2.putText(frame, f"ID:{person.id} C:{person.confidence:.2f}",
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    def draw_info_panel(self, frame, detection_info, fps_history):
        """绘制信息面板"""
        height, width = frame.shape[:2]

        # 计算平均FPS
        avg_fps = np.mean(fps_history) if fps_history else 0

        # 信息文本
        info_texts = [
            f"FPS: {avg_fps:.1f}",
            f"People: {detection_info['persons_count']}",
            f"Process: {detection_info['processing_time_ms']:.1f}ms",
            f"Detector: {detection_info['detector_info']['name']}",
            f"Frame: {detection_info['frame_size']}",
            "",
            "Press 'q' to quit",
            "Press 's' to screenshot"
        ]

        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 200), (40, 40, 40), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # 绘制文本
        y_offset = 30
        for text in info_texts:
            if text:  # 非空行
                color = (0, 255, 0) if "FPS" in text else (255, 255, 255)
                cv2.putText(frame, text, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 20

    def save_performance_report(self, performance_results):
        """保存性能报告"""
        report_file = "performance_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🚀 姿态检测器性能测试报告\n")
            f.write("=" * 50 + "\n\n")

            for size_name, results in performance_results.items():
                f.write(f"📊 {size_name}:\n")
                f.write(f"   平均处理时间: {results['avg_time_ms']:.2f}ms\n")
                f.write(f"   预估FPS: {results['fps_estimate']:.1f}\n")
                f.write(f"   时间范围: {results['min_time_ms']:.2f}ms - {results['max_time_ms']:.2f}ms\n\n")

            f.write("测试完成时间: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")

        print(f"📄 性能报告已保存到: {report_file}")

    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始完整测试流程...")
        print("请确保摄像头可用且光线充足")

        # 测试序列
        tests = [
            ("基础导入", self.test_1_basic_import),
            ("MediaPipe检测器", self.test_2_mediapipe_detector),
            ("摄像头实时检测", self.test_3_webcam_detection),
            ("性能基准测试", self.test_4_performance_benchmark),
            ("检测器切换", self.test_5_detector_switching),
        ]

        passed_tests = 0

        for test_name, test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
                    print(f"✅ {test_name} 通过")
                else:
                    print(f"❌ {test_name} 失败")
            except KeyboardInterrupt:
                print(f"\n⏹️ 用户中断了 {test_name}")
                break
            except Exception as e:
                print(f"❌ {test_name} 异常: {e}")

        # 最终报告
        print("\n" + "=" * 60)
        print("🏁 测试总结")
        print(f"✅ 通过测试: {passed_tests}/{len(tests)}")
        print(f"📊 成功率: {passed_tests / len(tests) * 100:.1f}%")

        if passed_tests == len(tests):
            print("🎉 所有测试通过！检测器工作正常")
            return True
        else:
            print("⚠️ 部分测试失败，请检查问题")
            return False


def main():
    """主函数"""
    print("🎯 姿态检测器测试程序")
    print("确保requirements.txt中的依赖已安装:")
    print("   pip install opencv-python mediapipe numpy")
    print()

    # 创建测试器
    tester = PoseDetectorTester()

    # 询问测试模式
    print("选择测试模式:")
    print("1. 完整测试 (推荐)")
    print("2. 仅基础测试 (快速)")
    print("3. 仅摄像头测试")
    print("4. 仅性能测试")

    try:
        choice = input("请输入选择 (1-4): ").strip()

        if choice == '1':
            tester.run_all_tests()
        elif choice == '2':
            tester.test_1_basic_import()
            tester.test_2_mediapipe_detector()
        elif choice == '3':
            tester.test_3_webcam_detection()
        elif choice == '4':
            tester.test_4_performance_benchmark()
        else:
            print("无效选择，运行完整测试")
            tester.run_all_tests()

    except KeyboardInterrupt:
        print("\n👋 测试被用户中断")
    except Exception as e:
        print(f"❌ 测试程序异常: {e}")

    print("\n🏁 测试程序结束")


if __name__ == "__main__":
    main()
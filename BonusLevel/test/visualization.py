# 🎨 完整可视化评估系统

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from collections import deque, defaultdict
import time
import tkinter as tk
from tkinter import ttk
import threading
from datetime import datetime
import json


class VisualizationSystem:
    """
    综合可视化评估系统

    功能：
    1. 实时姿态显示 - 骨架图、关键点、边界框
    2. 性能监控面板 - FPS、延迟、错误率
    3. 问题解决效果对比 - Before/After
    4. 相似度评分可视化 - 雷达图、趋势图
    5. 调试信息显示 - 问题检测、修复状态
    """

    def __init__(self):
        self.setup_colors()
        self.setup_fonts()
        self.setup_monitoring()

        # 可视化模式
        self.visualization_modes = {
            'basic': True,  # 基础骨架显示
            'debug': False,  # 调试信息显示
            'comparison': False,  # 对比模式
            'metrics': True,  # 性能指标
            'similarity': True  # 相似度显示
        }

        print("🎨 VisualizationSystem initialized")

    def setup_colors(self):
        """设置配色方案"""
        self.colors = {
            # 骨架颜色
            'skeleton_good': (0, 255, 0),  # 绿色 - 正常
            'skeleton_warning': (0, 255, 255),  # 黄色 - 警告
            'skeleton_error': (0, 0, 255),  # 红色 - 错误

            # 关键点颜色
            'keypoint_high_conf': (0, 255, 0),  # 高置信度
            'keypoint_med_conf': (0, 255, 255),  # 中置信度
            'keypoint_low_conf': (0, 0, 255),  # 低置信度
            'keypoint_predicted': (255, 0, 255),  # 预测点

            # UI颜色
            'text_primary': (255, 255, 255),  # 主要文本
            'text_secondary': (200, 200, 200),  # 次要文本
            'background': (0, 0, 0),  # 背景
            'panel_bg': (40, 40, 40),  # 面板背景

            # 状态颜色
            'status_good': (0, 255, 0),
            'status_warning': (0, 255, 255),
            'status_error': (0, 0, 255),
        }

    def setup_fonts(self):
        """设置字体"""
        self.fonts = {
            'small': cv2.FONT_HERSHEY_SIMPLEX,
            'medium': cv2.FONT_HERSHEY_SIMPLEX,
            'large': cv2.FONT_HERSHEY_SIMPLEX,
            'size_small': 0.5,
            'size_medium': 0.7,
            'size_large': 1.0
        }

    def setup_monitoring(self):
        """设置监控数据结构"""
        self.metrics_history = {
            'fps': deque(maxlen=300),  # 10秒历史(30fps)
            'processing_time': deque(maxlen=300),
            'detection_count': deque(maxlen=300),
            'similarity_scores': deque(maxlen=300),
            'error_counts': deque(maxlen=300),
        }

        self.problem_status = {
            'question1_proximity': {'count': 0, 'fixed': 0},
            'question2_multipeople': {'count': 0, 'fixed': 0},
            'question3_lag': {'count': 0, 'fixed': 0},
            'question4_occlusion': {'count': 0, 'fixed': 0},
        }

    # =============================================================================
    # 1. 实时姿态可视化
    # =============================================================================

    def draw_pose_with_status(self, frame: np.ndarray, persons: list,
                              debug_info: dict = None) -> np.ndarray:
        """
        绘制带状态信息的姿态

        特色功能：
        - 颜色编码显示问题状态
        - 实时调试信息叠加
        - 多人ID稳定性显示
        """
        vis_frame = frame.copy()

        for person in persons:
            person_id = person.get('id', 0)
            keypoints = person.get('keypoints', [])
            confidence = person.get('confidence', 0)

            # 根据人员状态选择颜色
            skeleton_color = self._get_person_status_color(person, debug_info)

            # 绘制骨架
            self._draw_skeleton(vis_frame, keypoints, skeleton_color)

            # 绘制关键点（根据置信度着色）
            self._draw_keypoints_with_confidence(vis_frame, keypoints)

            # 绘制边界框和ID
            self._draw_person_info(vis_frame, person, skeleton_color)

            # 绘制调试信息
            if self.visualization_modes['debug'] and debug_info:
                self._draw_debug_info(vis_frame, person, debug_info)

        # 绘制整体状态面板
        if self.visualization_modes['metrics']:
            self._draw_metrics_panel(vis_frame)

        return vis_frame

    def _get_person_status_color(self, person: dict, debug_info: dict) -> tuple:
        """根据人员状态获取颜色"""
        if not debug_info:
            return self.colors['skeleton_good']

        person_id = person.get('id')
        issues = debug_info.get('issues', {}).get(person_id, {})

        if issues.get('critical_issues', 0) > 0:
            return self.colors['skeleton_error']
        elif issues.get('warnings', 0) > 0:
            return self.colors['skeleton_warning']
        else:
            return self.colors['skeleton_good']

    def _draw_skeleton(self, frame: np.ndarray, keypoints: list, color: tuple):
        """绘制骨架连接"""
        skeleton_connections = [
            (5, 6),  # 肩膀
            (5, 7), (7, 9),  # 左臂
            (6, 8), (8, 10),  # 右臂
            (5, 11), (6, 12), (11, 12),  # 躯干
            (11, 13), (13, 15),  # 左腿
            (12, 14), (14, 16),  # 右腿
        ]

        for start_idx, end_idx in skeleton_connections:
            if (start_idx < len(keypoints) and end_idx < len(keypoints)):
                start_kp = keypoints[start_idx]
                end_kp = keypoints[end_idx]

                if (start_kp.get('visible', False) and end_kp.get('visible', False) and
                        start_kp.get('confidence', 0) > 0.3 and end_kp.get('confidence', 0) > 0.3):
                    start_point = (int(start_kp['x']), int(start_kp['y']))
                    end_point = (int(end_kp['x']), int(end_kp['y']))

                    # 根据连接质量调整线条粗细
                    thickness = 2 if min(start_kp['confidence'], end_kp['confidence']) > 0.7 else 1
                    cv2.line(frame, start_point, end_point, color, thickness)

    def _draw_keypoints_with_confidence(self, frame: np.ndarray, keypoints: list):
        """根据置信度绘制关键点"""
        for i, kp in enumerate(keypoints):
            if not kp.get('visible', False):
                continue

            x, y = int(kp['x']), int(kp['y'])
            confidence = kp.get('confidence', 0)
            is_predicted = kp.get('predicted', False)

            # 根据置信度和状态选择颜色
            if is_predicted:
                color = self.colors['keypoint_predicted']
                radius = 3
            elif confidence > 0.7:
                color = self.colors['keypoint_high_conf']
                radius = 4
            elif confidence > 0.4:
                color = self.colors['keypoint_med_conf']
                radius = 3
            else:
                color = self.colors['keypoint_low_conf']
                radius = 2

            cv2.circle(frame, (x, y), radius, color, -1)

            # 绘制关键点ID（调试模式）
            if self.visualization_modes['debug']:
                cv2.putText(frame, str(i), (x + 5, y - 5),
                            self.fonts['small'], self.fonts['size_small'],
                            self.colors['text_secondary'], 1)

    def _draw_person_info(self, frame: np.ndarray, person: dict, color: tuple):
        """绘制人员信息"""
        bbox = person.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 绘制人员ID和置信度
            person_id = person.get('id', 'Unknown')
            confidence = person.get('confidence', 0)
            track_hits = person.get('track_hits', 0)

            info_text = f"ID:{person_id} C:{confidence:.2f}"
            if track_hits > 0:
                info_text += f" H:{track_hits}"

            # 背景矩形
            text_size = cv2.getTextSize(info_text, self.fonts['small'],
                                        self.fonts['size_small'], 1)[0]
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_size[0] + 10, y1),
                          self.colors['panel_bg'], -1)

            # 文本
            cv2.putText(frame, info_text, (x1 + 5, y1 - 8),
                        self.fonts['small'], self.fonts['size_small'],
                        self.colors['text_primary'], 1)

    # =============================================================================
    # 2. 性能监控面板
    # =============================================================================

    def _draw_metrics_panel(self, frame: np.ndarray):
        """绘制性能指标面板"""
        height, width = frame.shape[:2]
        panel_width = 250
        panel_height = 200

        # 面板背景
        panel_x = width - panel_width - 10
        panel_y = 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      self.colors['panel_bg'], -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)

        # 绘制指标
        y_offset = panel_y + 25
        metrics = self._get_current_metrics()

        # FPS
        fps_color = self._get_status_color(metrics['fps'], 25, 20)
        cv2.putText(frame, f"FPS: {metrics['fps']:.1f}",
                    (panel_x + 10, y_offset), self.fonts['medium'],
                    self.fonts['size_medium'], fps_color, 2)
        y_offset += 30

        # 处理时间
        proc_color = self._get_status_color(50 - metrics['processing_time_ms'], 30, 20)
        cv2.putText(frame, f"Process: {metrics['processing_time_ms']:.1f}ms",
                    (panel_x + 10, y_offset), self.fonts['small'],
                    self.fonts['size_small'], proc_color, 1)
        y_offset += 25

        # 检测人数
        cv2.putText(frame, f"People: {metrics['people_count']}",
                    (panel_x + 10, y_offset), self.fonts['small'],
                    self.fonts['size_small'], self.colors['text_primary'], 1)
        y_offset += 25

        # 问题统计
        cv2.putText(frame, "Issues Fixed:",
                    (panel_x + 10, y_offset), self.fonts['small'],
                    self.fonts['size_small'], self.colors['text_secondary'], 1)
        y_offset += 20

        for problem, stats in self.problem_status.items():
            if stats['count'] > 0:
                fix_rate = stats['fixed'] / stats['count'] * 100
                color = self.colors['status_good'] if fix_rate > 80 else self.colors['status_warning']
                problem_short = problem.replace('question', 'Q').replace('_', ' ')
                cv2.putText(frame, f"{problem_short}: {fix_rate:.0f}%",
                            (panel_x + 15, y_offset), self.fonts['small'],
                            0.4, color, 1)
                y_offset += 18

    def _get_status_color(self, value: float, good_threshold: float, bad_threshold: float) -> tuple:
        """根据数值获取状态颜色"""
        if value >= good_threshold:
            return self.colors['status_good']
        elif value >= bad_threshold:
            return self.colors['status_warning']
        else:
            return self.colors['status_error']

    # =============================================================================
    # 3. 相似度可视化
    # =============================================================================

    def draw_similarity_display(self, frame: np.ndarray, similarity_scores: dict):
        """绘制相似度显示"""
        if not similarity_scores:
            return frame

        height, width = frame.shape[:2]

        # 主要相似度分数 - 大字显示
        total_score = similarity_scores.get('total_score', 0)
        score_text = f"{total_score:.1f}"

        # 获取分数颜色
        if total_score >= 80:
            score_color = self.colors['status_good']
        elif total_score >= 60:
            score_color = self.colors['status_warning']
        else:
            score_color = self.colors['status_error']

        # 绘制大分数
        font_scale = 2.0
        thickness = 3
        text_size = cv2.getTextSize(score_text, self.fonts['large'], font_scale, thickness)[0]

        # 位置：右上角
        score_x = width - text_size[0] - 20
        score_y = 60

        # 背景
        cv2.rectangle(frame, (score_x - 10, score_y - text_size[1] - 10),
                      (score_x + text_size[0] + 10, score_y + 10),
                      self.colors['panel_bg'], -1)

        # 分数文本
        cv2.putText(frame, score_text, (score_x, score_y),
                    self.fonts['large'], font_scale, score_color, thickness)

        # 分数描述
        description = self._get_score_description(total_score)
        cv2.putText(frame, description, (score_x - 20, score_y + 25),
                    self.fonts['small'], self.fonts['size_small'],
                    self.colors['text_primary'], 1)

        # 详细分数条
        self._draw_score_bars(frame, similarity_scores, score_x - 150, score_y + 50)

        return frame

    def _draw_score_bars(self, frame: np.ndarray, scores: dict, x: int, y: int):
        """绘制详细分数条"""
        bar_width = 120
        bar_height = 8
        categories = [
            ('pose_similarity', 'Pose'),
            ('angle_similarity', 'Angle'),
            ('proportion_similarity', 'Proportion'),
            ('symmetry_score', 'Symmetry')
        ]

        for i, (key, label) in enumerate(categories):
            score = scores.get(key, 0)
            bar_y = y + i * 20

            # 背景条
            cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height),
                          (50, 50, 50), -1)

            # 分数条
            fill_width = int(bar_width * score / 100)
            color = self._get_status_color(score, 80, 60)
            cv2.rectangle(frame, (x, bar_y), (x + fill_width, bar_y + bar_height),
                          color, -1)

            # 标签
            cv2.putText(frame, f"{label}: {score:.0f}",
                        (x + bar_width + 10, bar_y + bar_height),
                        self.fonts['small'], 0.4, self.colors['text_primary'], 1)

    def _get_score_description(self, score: float) -> str:
        """获取分数描述"""
        if score >= 90:
            return "Perfect! 🌟"
        elif score >= 80:
            return "Excellent! 🎉"
        elif score >= 70:
            return "Good! 👍"
        elif score >= 60:
            return "Not bad! 👌"
        elif score >= 50:
            return "Keep trying! 💪"
        else:
            return "Practice more! 🔥"

    # =============================================================================
    # 4. 问题检测可视化
    # =============================================================================

    def update_problem_status(self, problem_type: str, detected: bool, fixed: bool = False):
        """更新问题状态"""
        if problem_type in self.problem_status:
            if detected:
                self.problem_status[problem_type]['count'] += 1
            if fixed:
                self.problem_status[problem_type]['fixed'] += 1

    def _draw_debug_info(self, frame: np.ndarray, person: dict, debug_info: dict):
        """绘制调试信息"""
        person_id = person.get('id')
        bbox = person.get('bbox', [])

        if len(bbox) != 4:
            return

        x1, y1, x2, y2 = map(int, bbox)
        issues = debug_info.get('issues', {}).get(person_id, {})

        # 在人员上方显示问题信息
        debug_y = y1 - 40

        for issue_type, details in issues.items():
            if details.get('detected', False):
                issue_text = f"{issue_type}: {details.get('severity', 'Unknown')}"
                cv2.putText(frame, issue_text, (x1, debug_y),
                            self.fonts['small'], 0.4,
                            self.colors['status_error'], 1)
                debug_y -= 15


# =============================================================================
# 5. 对比分析可视化
# =============================================================================

class ComparisonVisualizer:
    """对比分析可视化器"""

    def __init__(self):
        self.before_metrics = []
        self.after_metrics = []

    def create_before_after_comparison(self, save_path: str = "comparison_report.png"):
        """创建问题解决前后对比图"""

        # 模拟数据（实际使用时从日志读取）
        problems = ['Q1: Proximity', 'Q2: Multi-person', 'Q3: Lag', 'Q4: Occlusion']
        before_scores = [60, 40, 18, 30]  # 解决前指标
        after_scores = [85, 90, 29, 75]  # 解决后指标

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dance Detection - Problem Solutions Comparison', fontsize=16, fontweight='bold')

        # 1. 总体改善对比
        x = np.arange(len(problems))
        width = 0.35

        ax1.bar(x - width / 2, before_scores, width, label='Before', color='#ff6b6b', alpha=0.8)
        ax1.bar(x + width / 2, after_scores, width, label='After', color='#4ecdc4', alpha=0.8)
        ax1.set_xlabel('Problems')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Problem Solutions Effectiveness')
        ax1.set_xticks(x)
        ax1.set_xticklabels(problems, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 添加改善百分比标注
        for i, (before, after) in enumerate(zip(before_scores, after_scores)):
            improvement = ((after - before) / before) * 100
            ax1.text(i, max(before, after) + 2, f'+{improvement:.0f}%',
                     ha='center', va='bottom', fontweight='bold', color='green')

        # 2. FPS性能趋势
        time_points = np.linspace(0, 100, 100)
        fps_before = 15 + 3 * np.sin(time_points * 0.1) + np.random.normal(0, 1, 100)
        fps_after = 28 + 2 * np.sin(time_points * 0.05) + np.random.normal(0, 0.5, 100)

        ax2.plot(time_points, fps_before, label='Before (avg: 15.2 FPS)', color='#ff6b6b', alpha=0.7)
        ax2.plot(time_points, fps_after, label='After (avg: 28.5 FPS)', color='#4ecdc4', alpha=0.7)
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Target: 30 FPS')
        ax2.set_xlabel('Time (frames)')
        ax2.set_ylabel('FPS')
        ax2.set_title('Frame Rate Stability Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 相似度分布
        similarity_before = np.random.normal(45, 15, 1000)
        similarity_after = np.random.normal(78, 8, 1000)

        ax3.hist(similarity_before, bins=20, alpha=0.6, label='Before', color='#ff6b6b', density=True)
        ax3.hist(similarity_after, bins=20, alpha=0.6, label='After', color='#4ecdc4', density=True)
        ax3.set_xlabel('Similarity Score')
        ax3.set_ylabel('Density')
        ax3.set_title('Similarity Score Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 问题检测和修复统计
        categories = ['Detected', 'Fixed', 'Fix Rate']
        q1_stats = [100, 85, 85]
        q2_stats = [80, 72, 90]
        q3_stats = [150, 140, 93]
        q4_stats = [60, 45, 75]

        x = np.arange(len(categories))
        width = 0.2

        ax4.bar(x - 1.5 * width, q1_stats, width, label='Q1: Proximity', alpha=0.8)
        ax4.bar(x - 0.5 * width, q2_stats, width, label='Q2: Multi-person', alpha=0.8)
        ax4.bar(x + 0.5 * width, q3_stats, width, label='Q3: Lag', alpha=0.8)
        ax4.bar(x + 1.5 * width, q4_stats, width, label='Q4: Occlusion', alpha=0.8)

        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Count / Percentage')
        ax4.set_title('Problem Detection & Fix Statistics')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 对比分析图已保存到: {save_path}")


# =============================================================================
# 6. 实时监控GUI
# =============================================================================

class RealTimeMonitorGUI:
    """实时监控GUI面板"""

    def __init__(self, visualization_system):
        self.vis_sys = visualization_system
        self.root = tk.Tk()
        self.root.title("🎯 Dance Detection - Real-time Monitor")
        self.root.geometry("400x600")

        self.setup_gui()
        self.update_thread = None
        self.running = False

    def setup_gui(self):
        """设置GUI界面"""

        # 主标题
        title_label = tk.Label(self.root, text="🎯 Real-time Monitor",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # 性能指标区域
        self.setup_performance_section()

        # 问题状态区域
        self.setup_problem_status_section()

        # 相似度区域
        self.setup_similarity_section()

        # 控制按钮
        self.setup_control_buttons()

    def setup_performance_section(self):
        """性能指标区域"""
        perf_frame = tk.LabelFrame(self.root, text="📊 Performance Metrics",
                                   font=("Arial", 12, "bold"))
        perf_frame.pack(fill="x", padx=10, pady=5)

        # FPS
        self.fps_label = tk.Label(perf_frame, text="FPS: --",
                                  font=("Arial", 11))
        self.fps_label.pack(anchor="w", padx=10, pady=2)

        # 处理时间
        self.process_time_label = tk.Label(perf_frame, text="Processing: -- ms",
                                           font=("Arial", 11))
        self.process_time_label.pack(anchor="w", padx=10, pady=2)

        # 检测人数
        self.people_count_label = tk.Label(perf_frame, text="People Detected: --",
                                           font=("Arial", 11))
        self.people_count_label.pack(anchor="w", padx=10, pady=2)

        # FPS图表按钮
        self.fps_chart_btn = tk.Button(perf_frame, text="📈 Show FPS Chart",
                                       command=self.show_fps_chart)
        self.fps_chart_btn.pack(pady=5)

    def setup_problem_status_section(self):
        """问题状态区域"""
        problem_frame = tk.LabelFrame(self.root, text="🔧 Problem Solutions",
                                      font=("Arial", 12, "bold"))
        problem_frame.pack(fill="x", padx=10, pady=5)

        self.problem_labels = {}
        problems = [
            ("question1_proximity", "Q1: Proximity Issues"),
            ("question2_multipeople", "Q2: Multi-person Tracking"),
            ("question3_lag", "Q3: Frame Lag"),
            ("question4_occlusion", "Q4: Occlusion Handling")
        ]

        for key, label in problems:
            frame = tk.Frame(problem_frame)
            frame.pack(fill="x", padx=10, pady=2)

            tk.Label(frame, text=label + ":", font=("Arial", 10)).pack(side="left")
            status_label = tk.Label(frame, text="✅ 0/0 (0%)",
                                    font=("Arial", 10), fg="green")
            status_label.pack(side="right")

            self.problem_labels[key] = status_label

    def setup_similarity_section(self):
        """相似度区域"""
        sim_frame = tk.LabelFrame(self.root, text="🎯 Similarity Scores",
                                  font=("Arial", 12, "bold"))
        sim_frame.pack(fill="x", padx=10, pady=5)

        # 总分
        self.total_score_label = tk.Label(sim_frame, text="Total Score: --",
                                          font=("Arial", 12, "bold"))
        self.total_score_label.pack(pady=5)

        # 详细分数
        self.detail_scores = {}
        details = ["pose_similarity", "angle_similarity", "proportion_similarity", "symmetry_score"]

        for detail in details:
            label = tk.Label(sim_frame, text=f"{detail.replace('_', ' ').title()}: --",
                             font=("Arial", 10))
            label.pack(anchor="w", padx=10, pady=1)
            self.detail_scores[detail] = label

        # 分数历史图表按钮
        self.score_chart_btn = tk.Button(sim_frame, text="📊 Show Score History",
                                         command=self.show_score_history)
        self.score_chart_btn.pack(pady=5)

    def setup_control_buttons(self):
        """控制按钮"""
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=10)

        # 开始/停止监控
        self.start_btn = tk.Button(control_frame, text="▶️ Start Monitoring",
                                   command=self.start_monitoring, bg="lightgreen")
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = tk.Button(control_frame, text="⏹️ Stop Monitoring",
                                  command=self.stop_monitoring, bg="lightcoral",
                                  state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        # 生成报告
        self.report_btn = tk.Button(control_frame, text="📋 Generate Report",
                                    command=self.generate_report, bg="lightblue")
        self.report_btn.pack(side="left", padx=5)

    def start_monitoring(self):
        """开始监控"""
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()

        print("🚀 Real-time monitoring started")

    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        print("⏹️ Real-time monitoring stopped")

    def update_loop(self):
        """更新循环"""
        while self.running:
            try:
                # 获取最新指标
                metrics = self.vis_sys._get_current_metrics()

                # 更新GUI
                self.root.after(0, self.update_gui, metrics)

                time.sleep(0.1)  # 100ms更新一次
            except Exception as e:
                print(f"❌ Monitor update error: {e}")
                break

    def update_gui(self, metrics):
        """更新GUI显示"""
        try:
            # 更新性能指标
            fps = metrics.get('fps', 0)
            fps_color = "green" if fps >= 25 else "orange" if fps >= 20 else "red"
            self.fps_label.config(text=f"FPS: {fps:.1f}", fg=fps_color)

            proc_time = metrics.get('processing_time_ms', 0)
            proc_color = "green" if proc_time <= 30 else "orange" if proc_time <= 50 else "red"
            self.process_time_label.config(text=f"Processing: {proc_time:.1f} ms", fg=proc_color)

            self.people_count_label.config(text=f"People Detected: {metrics.get('people_count', 0)}")

            # 更新问题状态
            for problem_key, label_widget in self.problem_labels.items():
                stats = self.vis_sys.problem_status.get(problem_key, {'count': 0, 'fixed': 0})
                if stats['count'] > 0:
                    fix_rate = (stats['fixed'] / stats['count']) * 100
                    color = "green" if fix_rate >= 80 else "orange" if fix_rate >= 60 else "red"
                    text = f"✅ {stats['fixed']}/{stats['count']} ({fix_rate:.0f}%)"
                else:
                    color = "gray"
                    text = "✅ 0/0 (0%)"

                label_widget.config(text=text, fg=color)

            # 更新相似度（如果有数据）
            if hasattr(self, 'latest_similarity'):
                similarity = self.latest_similarity
                total_score = similarity.get('total_score', 0)
                score_color = "green" if total_score >= 80 else "orange" if total_score >= 60 else "red"
                self.total_score_label.config(text=f"Total Score: {total_score:.1f}", fg=score_color)

                for key, label_widget in self.detail_scores.items():
                    score = similarity.get(key, 0)
                    label_widget.config(text=f"{key.replace('_', ' ').title()}: {score:.1f}")

        except Exception as e:
            print(f"❌ GUI update error: {e}")

    def show_fps_chart(self):
        """显示FPS历史图表"""
        self.create_fps_chart()

    def show_score_history(self):
        """显示分数历史图表"""
        self.create_score_chart()

    def create_fps_chart(self):
        """创建FPS图表"""
        plt.figure(figsize=(12, 6))

        if len(self.vis_sys.metrics_history['fps']) > 0:
            fps_data = list(self.vis_sys.metrics_history['fps'])
            time_points = np.arange(len(fps_data))

            plt.subplot(1, 2, 1)
            plt.plot(time_points, fps_data, label='Actual FPS', color='#4ecdc4', linewidth=2)
            plt.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Target: 30 FPS')
            plt.axhline(y=np.mean(fps_data), color='red', linestyle=':', alpha=0.7,
                        label=f'Average: {np.mean(fps_data):.1f} FPS')
            plt.xlabel('Time (frames)')
            plt.ylabel('FPS')
            plt.title('Real-time FPS Performance')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # FPS分布直方图
            plt.subplot(1, 2, 2)
            plt.hist(fps_data, bins=20, alpha=0.7, color='#4ecdc4', edgecolor='black')
            plt.axvline(x=np.mean(fps_data), color='red', linestyle='--',
                        label=f'Mean: {np.mean(fps_data):.1f}')
            plt.axvline(x=np.median(fps_data), color='orange', linestyle='--',
                        label=f'Median: {np.median(fps_data):.1f}')
            plt.xlabel('FPS')
            plt.ylabel('Frequency')
            plt.title('FPS Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No FPS data available', ha='center', va='center',
                     transform=plt.gca().transAxes, fontsize=16)

        plt.tight_layout()
        plt.show()

    def create_score_chart(self):
        """创建分数历史图表"""
        plt.figure(figsize=(12, 8))

        if len(self.vis_sys.metrics_history['similarity_scores']) > 0:
            # 这里需要从历史数据中提取分数信息
            # 简化示例
            time_points = np.arange(100)
            total_scores = 70 + 10 * np.sin(time_points * 0.1) + np.random.normal(0, 3, 100)

            plt.subplot(2, 2, 1)
            plt.plot(time_points, total_scores, label='Total Score', linewidth=2)
            plt.xlabel('Time (frames)')
            plt.ylabel('Score')
            plt.title('Similarity Score Trend')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 其他子图...

        plt.tight_layout()
        plt.show()

    def generate_report(self):
        """生成评估报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_data = {
            'timestamp': timestamp,
            'performance_metrics': self.vis_sys._get_current_metrics(),
            'problem_status': self.vis_sys.problem_status,
            'session_duration': time.time() - getattr(self, 'session_start', time.time())
        }

        report_file = f"dance_detection_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"📋 评估报告已生成: {report_file}")

        # 创建可视化报告
        comparison_viz = ComparisonVisualizer()
        comparison_viz.create_before_after_comparison(f"comparison_{timestamp}.png")

    def run(self):
        """运行GUI"""
        self.session_start = time.time()
        print("🖥️ Real-time monitor GUI started")
        self.root.mainloop()

    # =============================================================================
    # 7. 辅助方法
    # =============================================================================

    def _get_current_metrics(self) -> dict:
        """获取当前性能指标"""
        return {
            'fps': np.mean(self.metrics_history['fps']) if self.metrics_history['fps'] else 0,
            'processing_time_ms': np.mean(self.metrics_history['processing_time']) * 1000 if self.metrics_history[
                'processing_time'] else 0,
            'people_count': self.metrics_history['detection_count'][-1] if self.metrics_history[
                'detection_count'] else 0,
            'avg_similarity': np.mean(self.metrics_history['similarity_scores']) if self.metrics_history[
                'similarity_scores'] else 0,
        }

    def update_metrics(self, fps: float, processing_time: float, people_count: int, similarity: float = None):
        """更新性能指标"""
        self.metrics_history['fps'].append(fps)
        self.metrics_history['processing_time'].append(processing_time)
        self.metrics_history['detection_count'].append(people_count)
        if similarity is not None:
            self.metrics_history['similarity_scores'].append(similarity)


# =============================================================================
# 8. 使用示例
# =============================================================================

def demo_visualization_system():
    """可视化系统使用示例"""

    # 创建可视化系统
    vis_sys = VisualizationSystem()

    # 启动实时监控GUI
    monitor_gui = RealTimeMonitorGUI(vis_sys)

    # 在另一个线程中运行监控
    monitor_thread = threading.Thread(target=monitor_gui.run, daemon=True)
    monitor_thread.start()

    # 模拟数据更新
    for i in range(1000):
        # 模拟性能数据
        fps = 30 + np.random.normal(0, 2)
        processing_time = 0.03 + np.random.normal(0, 0.01)
        people_count = np.random.randint(1, 4)
        similarity = 75 + np.random.normal(0, 10)

        # 更新指标
        vis_sys.update_metrics(fps, processing_time, people_count, similarity)

        # 模拟问题检测
        if i % 20 == 0:
            problem_type = np.random.choice(list(vis_sys.problem_status.keys()))
            vis_sys.update_problem_status(problem_type, detected=True, fixed=np.random.choice([True, False]))

        time.sleep(0.1)

    print("🎨 Visualization system demo completed")


if __name__ == "__main__":
    demo_visualization_system()
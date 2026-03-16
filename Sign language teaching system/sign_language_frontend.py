import sys
import os
import random
import cv2
import numpy as np

# 设置QT_PLUGIN_PATH环境变量
qt_plugins_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8_env', 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins')
if os.path.exists(qt_plugins_path):
    os.environ['QT_PLUGIN_PATH'] = qt_plugins_path
else:
    # 尝试其他可能的路径
    for root, dirs, files in os.walk('.'):
        if 'plugins' in dirs and 'platforms' in os.listdir(os.path.join(root, 'plugins')):
            os.environ['QT_PLUGIN_PATH'] = os.path.join(root, 'plugins')
            break

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QStatusBar, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap, QImage, QIcon

# 导入YOLO检测器
class YoloDetector:
    def __init__(self):
        # 初始化YOLO模型
        self.is_running = False
        # 模拟检测功能
        self.use_mock = False
        # 检测脚本路径
        self.detect_script = os.path.join(os.getcwd(), 'detect.py')
        # 临时图像路径
        self.temp_image = os.path.join(os.getcwd(), 'temp_frame.png')
        
        # 检查detect.py脚本是否存在
        if not os.path.exists(self.detect_script):
            print(f"Detect script not found: {self.detect_script}")
            self.use_mock = True
        else:
            print("Using external detect.py script for Wave-YOLO detection")
    
    def detect_frame(self, frame):
        """
        检测单帧图像
        :param frame: OpenCV格式的图像（BGR）
        :return: tuple: (检测结果图像, 识别的手语字母)
        """
        if not self.is_running:
            return frame, ""
        
        # 使用模拟检测
        if self.use_mock:
            # 随机生成一个字母作为检测结果
            import string
            recognized_letter = random.choice(string.ascii_lowercase)
            # 在图像上绘制检测结果
            frame_with_result = frame.copy()
            cv2.putText(frame_with_result, f"Detected: {recognized_letter}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame_with_result, recognized_letter
        
        # 使用外部检测脚本
        try:
            # 保存临时图像
            cv2.imwrite(self.temp_image, frame)
            
            # 调用检测脚本
            import subprocess
            result = subprocess.run(
                ['python', self.detect_script, self.temp_image],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            # 解析检测结果
            recognized_letter = result.stdout.strip()
            
            # 在图像上绘制检测结果
            frame_with_result = frame.copy()
            if recognized_letter:
                cv2.putText(frame_with_result, f"Detected: {recognized_letter}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return frame_with_result, recognized_letter
        except Exception as e:
            print(f"Error during detection: {e}")
            return frame, ""
    
    def start_detection(self):
        """
        开始检测
        """
        self.is_running = True
    
    def stop_detection(self):
        """
        停止检测
        """
        self.is_running = False

class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_letter = ""
        self.recognized_letter = ""
        # 初始化摄像头
        self.capture = None
        self.is_capturing = False
        # 初始化YOLO检测器
        self.yolo_detector = YoloDetector()
        # 创建定时器，用于更新摄像头画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        # 摄像头分辨率
        self.camera_resolution = (640, 480)
        self.detection_resolution = (320, 240)
        # 检测频率控制
        self.detection_counter = 0
        self.detection_interval = 30
        # 更新状态栏显示检测方式
        if self.yolo_detector.use_mock:
            self.statusBar.showMessage("Running in demo mode (mock detection)")
        else:
            self.statusBar.showMessage("Using Wave-YOLO model for detection")
    
    def init_ui(self):
        # 设置窗口基本属性
        self.setWindowTitle("Sign Language Recognition System")
        self.setGeometry(100, 100, 1000, 700)
        self.setMinimumSize(1000, 700)
        self.setMaximumSize(1000, 700)
        
        # 设置窗口图标
        icon_path = os.path.join(os.getcwd(), 'handsreco.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            print(f"Set window icon to: {icon_path}")
        else:
            print(f"Icon file not found: {icon_path}")
    
        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        # 创建中央部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 设置全局字体
        app_font = QFont("Arial", 11)
        QApplication.setFont(app_font)
        
        # 标题
        title = QLabel("Sign Language Recognition System")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # 中间布局：左侧检测结果，右侧检测框
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(20)
        
        # 左侧：检测结果区域
        self.detection_result_group = QGroupBox("Detection Results")
        self.detection_result_group.setFixedSize(450, 450)
        self.detection_result_layout = QVBoxLayout(self.detection_result_group)
        
        # 显示字母区域
        self.letter_display = QGroupBox("Display Letter")
        self.letter_display.setFixedSize(400, 150)
        self.letter_display_layout = QVBoxLayout(self.letter_display)
        self.letter_label = QLabel()
        self.letter_label.setFont(QFont("Arial", 36, QFont.Bold))
        self.letter_label.setAlignment(Qt.AlignCenter)
        self.letter_display_layout.addWidget(self.letter_label)
        self.detection_result_layout.addWidget(self.letter_display, 0, Qt.AlignCenter)
        
        # 下方布局：提示正确/错误和正确示例
        bottom_left_layout = QHBoxLayout()
        bottom_left_layout.setSpacing(20)
        
        # 提示正确/错误区域
        self.correctness_group = QGroupBox("Correctness")
        self.correctness_group.setFixedSize(180, 150)
        self.correctness_layout = QVBoxLayout(self.correctness_group)
        self.correctness_label = QLabel()
        self.correctness_label.setFont(QFont("Arial", 18))
        self.correctness_label.setAlignment(Qt.AlignCenter)
        self.correctness_layout.addWidget(self.correctness_label)
        bottom_left_layout.addWidget(self.correctness_group)
        
        # 正确示例区域
        self.example_group = QGroupBox("Correct Example")
        self.example_group.setFixedSize(180, 150)
        self.example_layout = QVBoxLayout(self.example_group)
        self.example_label = QLabel()
        self.example_label.setAlignment(Qt.AlignCenter)
        self.example_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }")
        self.example_layout.addWidget(self.example_label)
        bottom_left_layout.addWidget(self.example_group)
        
        self.detection_result_layout.addLayout(bottom_left_layout)
        middle_layout.addWidget(self.detection_result_group)
        
        # 右侧：检测框（链接摄像头）
        self.detection_frame_group = QGroupBox("Detection Frame (Camera)")
        self.detection_frame_group.setFixedSize(450, 450)
        self.detection_frame_layout = QVBoxLayout(self.detection_frame_group)
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("QLabel { background-color: #000; color: white; border: 1px solid #ccc; }")
        self.camera_label.setText("Camera not started")
        self.camera_label.setFixedSize(400, 300)
        self.detection_frame_layout.addWidget(self.camera_label, 0, Qt.AlignCenter)
        middle_layout.addWidget(self.detection_frame_group)
        
        main_layout.addLayout(middle_layout)
        
        # 底部按钮布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(30)
        
        # 开始按钮
        self.start_button = QPushButton("Start")
        self.start_button.setFont(QFont("Arial", 16, QFont.Bold))
        self.start_button.setFixedSize(120, 50)
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 5px; }")
        self.start_button.clicked.connect(self.start_detection)
        button_layout.addWidget(self.start_button)
        
        # 中止按钮
        self.abort_button = QPushButton("Stop")
        self.abort_button.setFont(QFont("Arial", 16, QFont.Bold))
        self.abort_button.setFixedSize(120, 50)
        self.abort_button.setStyleSheet("QPushButton { background-color: #ff9800; color: white; border-radius: 5px; }")
        self.abort_button.clicked.connect(self.abort_detection)
        button_layout.addWidget(self.abort_button)
        
        # 生成新的字母按钮
        self.new_letter_button = QPushButton("New Letter")
        self.new_letter_button.setFont(QFont("Arial", 16, QFont.Bold))
        self.new_letter_button.setFixedSize(180, 50)
        self.new_letter_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; border-radius: 5px; }")
        self.new_letter_button.clicked.connect(self.generate_new_letter)
        button_layout.addWidget(self.new_letter_button)
        
        button_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(button_layout)
        
        # 强制布局更新
        central_widget.adjustSize()
        self.adjustSize()
    
    def start_detection(self):
        """Start detection"""
        # Generate random letter
        self.generate_new_letter()
        
        # Start camera
        if not self.is_capturing:
            # Try different camera indices
            for i in range(3):
                self.capture = cv2.VideoCapture(i)
                if self.capture.isOpened():
                    break
            
            if not self.capture or not self.capture.isOpened():
                QMessageBox.critical(self, "Error", "Failed to open camera")
                return
            
            # Set camera resolution
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])
            
            self.is_capturing = True
            # Start YOLO detection
            self.yolo_detector.start_detection()
            # Start timer to update frame every 30ms
            self.timer.start(30)
        
        self.statusBar.showMessage("Detection started")
    
    def abort_detection(self):
        """Stop detection"""
        if self.is_capturing:
            self.timer.stop()
            self.capture.release()
            self.is_capturing = False
            self.yolo_detector.stop_detection()
            self.camera_label.setText("Camera stopped")
        
        self.statusBar.showMessage("Detection stopped")
    
    def generate_new_letter(self):
        """Generate random letter"""
        # Generate random letter
        import string
        self.current_letter = random.choice(string.ascii_lowercase)
        self.letter_label.setText(self.current_letter.upper())
        
        # Display correct example image
        self.display_example_image(self.current_letter)
        
        # Clear correctness feedback
        self.correctness_label.setText("")
        
        self.statusBar.showMessage(f"New letter generated: {self.current_letter}")
    
    def display_example_image(self, letter):
        """Display gesture image for the letter"""
        # Load image from c:\Users\Admin\Desktop\正确手势 directory
        img_path = os.path.join("c:", "Users", "Admin", "Desktop", "正确手势", f"{letter}.png")
        
        # Try different case combinations
        if not os.path.exists(img_path):
            img_path = os.path.join("c:", "Users", "Admin", "Desktop", "正确手势", f"{letter.upper()}.png")
        
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            # Resize image to fit label
            scaled_pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.example_label.setPixmap(scaled_pixmap)
        else:
            self.example_label.setText(f"No image for {letter}")
    
    def update_frame(self):
        """Update camera frame"""
        if not self.is_capturing or self.capture is None:
            return
        
        ret, frame = self.capture.read()
        if not ret:
            return
        
        # Default to original frame
        detected_frame = frame
        recognized_letter = ""
        
        # Control detection frequency
        self.detection_counter += 1
        
        # Resize image for display
        resized_frame = cv2.resize(frame, self.detection_resolution, interpolation=cv2.INTER_LINEAR)
        
        # Only call YOLO detection when detection frequency condition is met
        if self.detection_counter % self.detection_interval == 0:
            # Use YOLO detector to detect frame
            detected_frame, recognized_letter = self.yolo_detector.detect_frame(resized_frame)
            
            # Resize detection result back to original resolution for display
            detected_frame = cv2.resize(detected_frame, self.camera_resolution, interpolation=cv2.INTER_LINEAR)
            
            # Update recognition result
            if recognized_letter:
                self.recognized_letter = recognized_letter.lower()
                # Compare recognition result with displayed letter
                self.check_correctness()
        else:
            # If detection frequency condition is not met, directly resize original frame to display resolution
            detected_frame = cv2.resize(frame, self.camera_resolution, interpolation=cv2.INTER_LINEAR)
        
        # 将OpenCV图像转换为QImage
        rgb_frame = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 显示图像
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.camera_label.setPixmap(scaled_pixmap)
    
    def check_correctness(self):
        """Check if recognition result is correct"""
        if self.current_letter and self.recognized_letter:
            if self.current_letter == self.recognized_letter:
                self.correctness_label.setText("Correct")
                self.correctness_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            else:
                self.correctness_label.setText("Incorrect")
                self.correctness_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
                # Display gesture image matching the displayed letter
                self.display_example_image(self.current_letter)
    
    def closeEvent(self, event):
        """Release resources when closing window"""
        if self.is_capturing:
            self.timer.stop()
            self.capture.release()
            self.yolo_detector.stop_detection()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())

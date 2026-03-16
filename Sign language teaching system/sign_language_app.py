import sys
import os
import random
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QStatusBar, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap, QImage

# 导入YOLO模型
from yolo import YOLO

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
        self.yolo = None
        # 创建定时器，用于更新摄像头画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        # 摄像头分辨率
        self.camera_resolution = (640, 480)
        self.detection_resolution = (320, 240)
        # 检测频率控制
        self.detection_counter = 0
        self.detection_interval = 30
        # 加载YOLO模型
        self.load_yolo_model()
    
    def load_yolo_model(self):
        """加载YOLO模型"""
        try:
            self.yolo = YOLO()
            self.statusBar.showMessage("Model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.statusBar.showMessage("Error loading model")
    
    def init_ui(self):
        # 设置窗口基本属性
        self.setWindowTitle("Sign Language Recognition System")
        self.setGeometry(100, 100, 1000, 700)
        self.setMinimumSize(1000, 700)
        self.setMaximumSize(1000, 700)
    
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
        self.detection_result_group = QGroupBox("检测结果")
        self.detection_result_group.setFixedSize(450, 450)
        self.detection_result_layout = QVBoxLayout(self.detection_result_group)
        
        # 显示字母区域
        self.letter_display = QGroupBox("显示字母")
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
        self.correctness_group = QGroupBox("提示正确/错误")
        self.correctness_group.setFixedSize(180, 150)
        self.correctness_layout = QVBoxLayout(self.correctness_group)
        self.correctness_label = QLabel()
        self.correctness_label.setFont(QFont("Arial", 18))
        self.correctness_label.setAlignment(Qt.AlignCenter)
        self.correctness_layout.addWidget(self.correctness_label)
        bottom_left_layout.addWidget(self.correctness_group)
        
        # 正确示例区域
        self.example_group = QGroupBox("正确示例")
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
        self.detection_frame_group = QGroupBox("检测框（链接摄像头）")
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
        self.start_button = QPushButton("开始")
        self.start_button.setFont(QFont("Arial", 16, QFont.Bold))
        self.start_button.setFixedSize(120, 50)
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border-radius: 5px; }")
        self.start_button.clicked.connect(self.start_detection)
        button_layout.addWidget(self.start_button)
        
        # 中止按钮
        self.abort_button = QPushButton("中止")
        self.abort_button.setFont(QFont("Arial", 16, QFont.Bold))
        self.abort_button.setFixedSize(120, 50)
        self.abort_button.setStyleSheet("QPushButton { background-color: #ff9800; color: white; border-radius: 5px; }")
        self.abort_button.clicked.connect(self.abort_detection)
        button_layout.addWidget(self.abort_button)
        
        # 生成新的字母按钮
        self.new_letter_button = QPushButton("生成新的字母")
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
        """开始检测"""
        # 生成随机字母
        self.generate_new_letter()
        
        # 启动摄像头
        if not self.is_capturing:
            # 尝试不同的摄像头索引
            for i in range(3):
                self.capture = cv2.VideoCapture(i)
                if self.capture.isOpened():
                    break
            
            if not self.capture or not self.capture.isOpened():
                QMessageBox.critical(self, "Error", "Failed to open camera")
                return
            
            # 设置摄像头分辨率
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_resolution[1])
            
            self.is_capturing = True
            # 启动定时器，每30毫秒更新一次画面
            self.timer.start(30)
        
        self.statusBar.showMessage("Detection started")
    
    def abort_detection(self):
        """中止检测"""
        if self.is_capturing:
            self.timer.stop()
            self.capture.release()
            self.is_capturing = False
            self.camera_label.setText("Camera stopped")
        
        self.statusBar.showMessage("Detection aborted")
    
    def generate_new_letter(self):
        """生成随机字母"""
        # 生成随机字母
        import string
        self.current_letter = random.choice(string.ascii_lowercase)
        self.letter_label.setText(self.current_letter.upper())
        
        # 显示正确示例图片
        self.display_example_image(self.current_letter)
        
        # 清空正确/错误提示
        self.correctness_label.setText("")
        
        self.statusBar.showMessage(f"New letter generated: {self.current_letter}")
    
    def display_example_image(self, letter):
        """显示字母对应的手势图片"""
        # 从c:\Users\Admin\Desktop\正确手势目录加载图片
        img_path = os.path.join("c:", "Users", "Admin", "Desktop", "正确手势", f"{letter}.png")
        
        # 尝试不同的大小写组合
        if not os.path.exists(img_path):
            img_path = os.path.join("c:", "Users", "Admin", "Desktop", "正确手势", f"{letter.upper()}.png")
        
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            # 调整图片大小以适应标签
            scaled_pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.example_label.setPixmap(scaled_pixmap)
        else:
            self.example_label.setText(f"No image for {letter}")
    
    def update_frame(self):
        """更新摄像头画面"""
        if not self.is_capturing or self.capture is None:
            return
        
        ret, frame = self.capture.read()
        if not ret:
            return
        
        # 检测频率控制
        self.detection_counter += 1
        
        # 缩放图像用于显示
        resized_frame = cv2.resize(frame, self.detection_resolution, interpolation=cv2.INTER_LINEAR)
        
        # 默认使用原始帧
        detected_frame = frame
        recognized_letter = ""
        
        # 只有满足检测频率条件时，才调用YOLO检测
        if self.detection_counter % self.detection_interval == 0 and self.yolo:
            try:
                # 格式转换：BGR to RGB
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                # 转成PIL Image
                from PIL import Image
                image = Image.fromarray(np.uint8(frame_rgb))
                # 进行检测
                result_image, recognized_letters = self.yolo.detect_image(image, return_results=True)
                # 格式转换：RGB to BGR
                detected_frame = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
                # 将检测结果放大回原始分辨率以便显示
                detected_frame = cv2.resize(detected_frame, self.camera_resolution, interpolation=cv2.INTER_LINEAR)
                
                # 更新识别结果
                if recognized_letters:
                    self.recognized_letter = recognized_letters.lower()
                    # 比对识别结果与显示字母
                    self.check_correctness()
            except Exception as e:
                print(f"Error during detection: {e}")
                # 如果检测失败，使用原始帧
                detected_frame = cv2.resize(frame, self.camera_resolution, interpolation=cv2.INTER_LINEAR)
        else:
            # 如果不满足检测频率条件，直接将原始帧缩放到显示分辨率
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
        """检查识别结果是否正确"""
        if self.current_letter and self.recognized_letter:
            if self.current_letter == self.recognized_letter:
                self.correctness_label.setText("正确")
                self.correctness_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            else:
                self.correctness_label.setText("错误")
                self.correctness_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
                # 显示与显示字母匹配的手势图片
                self.display_example_image(self.current_letter)
    
    def closeEvent(self, event):
        """关闭窗口时释放资源"""
        if self.is_capturing:
            self.timer.stop()
            self.capture.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())

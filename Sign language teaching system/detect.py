import sys
import os
import cv2
import numpy as np
from PIL import Image

# 确保使用当前目录的yolo模块
sys.path.insert(0, os.getcwd())
from yolo import YOLO

class Detector:
    def __init__(self):
        # 初始化YOLO模型
        self.yolo = YOLO()
        print("Successfully loaded YOLO model")
    
    def detect(self, image_path):
        """
        检测图像
        :param image_path: 图像路径
        :return: 识别的字母
        """
        try:
            # 加载图像
            image = Image.open(image_path)
            # 进行检测
            result_image, recognized_letter = self.yolo.detect_image(image, return_results=True)
            return recognized_letter
        except Exception as e:
            print(f"Error during detection: {e}")
            return ""

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    detector = Detector()
    result = detector.detect(image_path)
    print(result)

from PIL import Image
import os

# 源PNG文件路径
png_path = os.path.join(os.getcwd(), '{E1F7CE8F-A012-4BD6-A435-BC0BE31CADB1}.png')
# 目标ICO文件路径
ico_path = os.path.join(os.getcwd(), 'handsreco.ico')

try:
    # 打开PNG图像
    img = Image.open(png_path)
    print(f"Opened image: {png_path}")
    print(f"Image size: {img.size}")
    print(f"Image mode: {img.mode}")
    
    # 转换为ICO格式
    img.save(ico_path, format='ICO')
    print(f"Successfully converted to ICO: {ico_path}")
    
    # 验证ICO文件是否创建成功
    if os.path.exists(ico_path):
        print(f"ICO file created successfully. Size: {os.path.getsize(ico_path)} bytes")
    else:
        print("Failed to create ICO file")
        
except Exception as e:
    print(f"Error converting PNG to ICO: {e}")

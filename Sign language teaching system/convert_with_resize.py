from PIL import Image
import os

# 源PNG文件路径
png_path = os.path.join(os.getcwd(), '{E1F7CE8F-A012-4BD6-A435-BC0BE31CADB1}.png')
# 目标ICO文件路径
ico_path = os.path.join(os.getcwd(), 'handsreco.ico')

# 目标分辨率 (更高的分辨率)
target_size = (512, 512)

try:
    # 打开PNG图像
    img = Image.open(png_path)
    print(f"Opened image: {png_path}")
    print(f"Original image size: {img.size}")
    print(f"Image mode: {img.mode}")
    
    # 调整图像大小到目标分辨率
    resized_img = img.resize(target_size, Image.LANCZOS)
    print(f"Resized image to: {target_size}")
    
    # 转换为ICO格式
    resized_img.save(ico_path, format='ICO')
    print(f"Successfully converted to ICO: {ico_path}")
    
    # 验证ICO文件是否创建成功
    if os.path.exists(ico_path):
        print(f"ICO file created successfully. Size: {os.path.getsize(ico_path)} bytes")
        # 检查生成的ICO文件的大小
        ico_img = Image.open(ico_path)
        print(f"Generated ICO size: {ico_img.size}")
    else:
        print("Failed to create ICO file")
        
except Exception as e:
    print(f"Error converting PNG to ICO: {e}")

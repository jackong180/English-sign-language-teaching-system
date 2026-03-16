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
    print(f"Original image size: {img.size}")
    print(f"Image mode: {img.mode}")
    
    # 创建一个256x256的高质量图标
    # 首先创建一个白色背景的新图像
    high_quality_img = Image.new('RGBA', (256, 256), (255, 255, 255, 255))
    
    # 计算缩放比例，保持原始图像的宽高比
    img_width, img_height = img.size
    scale = min(256 / img_width, 256 / img_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    
    # 调整原始图像大小
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # 计算居中位置
    x_offset = (256 - new_width) // 2
    y_offset = (256 - new_height) // 2
    
    # 将调整大小后的图像粘贴到新图像的中心
    high_quality_img.paste(resized_img, (x_offset, y_offset), resized_img)
    
    # 保存为ICO格式
    high_quality_img.save(ico_path, format='ICO')
    print(f"Successfully converted to high-quality ICO: {ico_path}")
    print(f"ICO file size: {os.path.getsize(ico_path)} bytes")
    print(f"Created 256x256 high-quality icon")
    
    # 验证ICO文件是否创建成功
    if os.path.exists(ico_path):
        print(f"ICO file created successfully. Size: {os.path.getsize(ico_path)} bytes")
    else:
        print("Failed to create ICO file")
        
except Exception as e:
    print(f"Error converting PNG to ICO: {e}")
    import traceback
    traceback.print_exc()

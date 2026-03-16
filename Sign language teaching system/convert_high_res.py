from PIL import Image
import os
import io

# 源PNG文件路径
png_path = os.path.join(os.getcwd(), '{E1F7CE8F-A012-4BD6-A435-BC0BE31CADB1}.png')
# 目标ICO文件路径
ico_path = os.path.join(os.getcwd(), 'handsreco.ico')

# 尝试创建多尺寸的ICO文件，这样Windows可以根据需要选择最合适的尺寸
sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

try:
    # 打开PNG图像
    img = Image.open(png_path)
    print(f"Opened image: {png_path}")
    print(f"Original image size: {img.size}")
    print(f"Image mode: {img.mode}")
    
    # 创建一个内存中的ICO文件
    icon_bytes = io.BytesIO()
    
    # 准备不同尺寸的图像
    icon_images = []
    for size in sizes:
        # 调整图像大小到目标分辨率
        resized_img = img.resize(size, Image.LANCZOS)
        icon_images.append(resized_img)
        print(f"Resized image to: {size}")
    
    # 保存为多尺寸ICO文件
    icon_images[0].save(
        icon_bytes,
        format='ICO',
        sizes=[(size[0], size[1]) for size in sizes],
        append_images=icon_images[1:]
    )
    
    # 将内存中的ICO文件写入磁盘
    with open(ico_path, 'wb') as f:
        f.write(icon_bytes.getvalue())
    
    print(f"Successfully converted to multi-size ICO: {ico_path}")
    print(f"ICO file size: {os.path.getsize(ico_path)} bytes")
    print(f"Created ICO with sizes: {sizes}")
    
    # 验证ICO文件是否创建成功
    if os.path.exists(ico_path):
        print(f"ICO file created successfully. Size: {os.path.getsize(ico_path)} bytes")
        # 检查生成的ICO文件
        try:
            ico_img = Image.open(ico_path)
            print(f"Generated ICO format: {ico_img.format}")
        except Exception as e:
            print(f"Could not open generated ICO: {e}")
    else:
        print("Failed to create ICO file")
        
except Exception as e:
    print(f"Error converting PNG to ICO: {e}")
    import traceback
    traceback.print_exc()

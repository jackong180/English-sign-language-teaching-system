import torch
import sys
import os

# 添加WaveFormer路径到Python路径
sys.path.append('d:\\WaveFormer-main')

from nets.yolo import YoloBody

# 测试WaveFormer集成到YOLOv8
if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 初始化模型
    input_shape = [3, 640, 640]
    num_classes = 20  # VOC数据集的类别数
    phi = 's'  # 使用small版本
    
    print('Initializing YOLOv8 with WaveFormer backbone...')
    model = YoloBody(input_shape, num_classes, phi=phi, pretrained=False)
    model.to(device)
    
    # 创建测试输入
    input_tensor = torch.randn(1, 3, 640, 640).to(device)
    
    # 前向传播测试
    print('Testing forward pass...')
    with torch.no_grad():
        outputs = model(input_tensor)
    
    print('Forward pass successful!')
    print(f'Outputs type: {type(outputs)}')
    print(f'Number of outputs: {len(outputs)}')
    
    # 打印输出形状
    if isinstance(outputs, tuple):
        for i, output in enumerate(outputs):
            if isinstance(output, torch.Tensor):
                print(f'Output {i} shape: {output.shape}')
            elif isinstance(output, list):
                print(f'Output {i} is a list with {len(output)} tensors')
                for j, tensor in enumerate(output):
                    if isinstance(tensor, torch.Tensor):
                        print(f'  Tensor {j} shape: {tensor.shape}')
    
    print('WaveFormer integration test completed successfully!')

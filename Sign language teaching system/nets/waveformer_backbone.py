import torch
import torch.nn as nn
import torch.nn.functional as F
from .waveformer_dct import WaveFormer
from torch import nn

class WaveFormerBackbone(nn.Module):
    def __init__(self, input_shape, num_classes=None, phi='s', pretrained=False):
        super(WaveFormerBackbone, self).__init__()
        # WaveFormer配置参数
        depth_dict = {'n': [2, 2, 6, 2], 's': [2, 2, 9, 2], 'm': [3, 4, 18, 3], 'l': [3, 4, 27, 3], 'x': [3, 8, 36, 3]}
        dim_dict = {'n': [96, 192, 384, 768], 's': [96, 192, 384, 768], 'm': [128, 256, 512, 1024], 'l': [192, 384, 768, 1536], 'x': [256, 512, 1024, 2048]}
        
        depths = depth_dict[phi]
        dims = dim_dict[phi]
        
        # 初始化WaveFormer
        self.waveformer = WaveFormer(
            patch_size=4,
            in_chans=3,
            num_classes=num_classes if num_classes is not None else 1000,
            depths=depths,
            dims=dims,
            drop_path_rate=0.1,
            img_size=input_shape[1],
            post_norm=True,
            layer_scale=1e-6
        )
        
        # 移除分类器，保留特征提取部分
        if hasattr(self.waveformer, 'classifier'):
            del self.waveformer.classifier
        
        # 输出特征通道映射
        self.feat_channels = {
            'n': [192, 384, 768],
            's': [192, 384, 768],
            'm': [256, 512, 1024],
            'l': [384, 768, 1536],
            'x': [512, 1024, 2048]
        }[phi]
        
    def forward(self, x):
        # WaveFormer前向传播
        x = self.waveformer.patch_embed(x)
        features = []
        
        # 动态调整freq_embed的大小以匹配当前输入
        B, C, H, W = x.shape
        
        for i, layer in enumerate(self.waveformer.layers):
            # 检查当前freq_embed的大小是否与输入匹配
            freq_embed = self.waveformer.freq_embed[i]
            if freq_embed.shape[:2] != (H, W):
                # 如果不匹配，调整freq_embed的大小
                freq_embed = F.interpolate(
                    freq_embed.permute(2, 0, 1).unsqueeze(0),  # [1, C, H, W]
                    size=(H, W),
                    mode='bicubic',
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0).contiguous()  # [H, W, C]
            
            x = layer(x, freq_embed)
            
            # 收集第1、2、3阶段的输出（对应YOLOv8的feat1, feat2, feat3）
            if i in [0, 1, 2]:
                features.append(x)
                
            # 计算下一层的预期大小
            if i < len(self.waveformer.layers) - 1:
                H = H // 2
                W = W // 2
        
        # 返回三个尺度的特征图
        feat1, feat2, feat3 = features
        return feat1, feat2, feat3
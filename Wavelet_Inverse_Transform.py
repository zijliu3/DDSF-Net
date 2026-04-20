import torch
import torch.nn as nn
from pytorch_wavelets import DWTInverse

class WITR(nn.Module):
    """
    输入：
        yL: [B, C, H, W]
        yH: [B, C*3, H, W]  <-- 你的增强模块输出格式
    输出：
        x_rec: [B, C, 2H, 2W]
    """
    def __init__(self):
        super(WITR, self).__init__()
        self.idwt = DWTInverse(mode='zero', wave='haar')

    def forward(self, yL, yH):
        B, C3, H, W = yH.shape
        C = C3 // 3  # 计算每个方向的通道数

        # reshape 成标准 IDWT 格式: [B, C, 3, H, W]
        yH = yH.view(B, C, 3, H, W)

        # 逆小波重建
        x_rec = self.idwt((yL, [yH]))
        return x_rec

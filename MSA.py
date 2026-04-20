import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveCombiner(nn.Module):
    """自适应融合模块，可学习融合高低特征"""
    def __init__(self):
        super().__init__()
        # 一个可学习标量参数 d，用于控制融合比例
        self.d = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, p, i):
        # p 和 i shape: [B, C, H, W]
        B, C, H, W = p.shape
        d = self.d.expand(B, C, H, W)
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i


class conv_block(nn.Module):
    """基础卷积模块，可选 BN/GN 和 ReLU"""
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)
        self.norm_type = norm_type
        self.act = activation
        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        elif self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class DPCF(nn.Module):
    """Detail-Preserving Contextual Fusion 模块"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.ac = AdaptiveCombiner()
        self.tail_conv = nn.Conv2d(in_features, out_features, kernel_size=1)

    def forward(self, x_low, x_high):
        B, C, H, W = x_low.shape

        # 检查通道是否可以被4整除
        assert C % 4 == 0, f"x_low channels ({C}) must be divisible by 4"

        # 分通道成4份
        x_low_chunks = torch.chunk(x_low, 4, dim=1)
        # 上采样高层特征到低层尺寸
        x_high = F.interpolate(x_high, size=(H, W), mode='bilinear', align_corners=True)
        x_high_chunks = torch.chunk(x_high, 4, dim=1)

        # 对每一份进行自适应融合
        out_chunks = [self.ac(l, h) for l, h in zip(x_low_chunks, x_high_chunks)]

        # 拼接并用1x1卷积整合通道
        x = torch.cat(out_chunks, dim=1)
        x = self.tail_conv(x)
        return x


if __name__ == "__main__":
    # 测试任意 H×W
    input1 = torch.randn(1, 32, 75, 50)  # 非方形输入
    input2 = torch.randn(1, 32, 20, 15)  # 高层低分辨率

    dpcf = DPCF(in_features=32, out_features=32)
    output = dpcf(input1, input2)

    print("input1 shape:", input1.shape)
    print("input2 shape:", input2.shape)
    print("output shape:", output.shape)

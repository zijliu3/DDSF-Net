from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 通道独立大核卷积（DConv7）
# -----------------------------
class DConv7(nn.Module):
    def __init__(self, f_number, padding_mode='reflect') -> None:
        super().__init__()
        self.dconv = nn.Conv2d(
            f_number, f_number, kernel_size=7, padding=3, groups=f_number, padding_mode=padding_mode
        )

    def forward(self, x):
        return self.dconv(x)

# -----------------------------
# SpeKAN 模块（替换 CID 中的 KAN/MLP）
# -----------------------------
class KANLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

class SpeKAN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # avg/max pool -> KANLinear -> KANLinear
        self.avg_kan01 = KANLinear(dim, dim)
        self.avg_kan02 = KANLinear(dim, dim)
        self.max_kan01 = KANLinear(dim, dim)
        self.max_kan02 = KANLinear(dim, dim)
        # 拼接 avg+max 再做一次映射
        self.sum_kan = KANLinear(dim * 2, dim)

    def forward(self, x):
        # 全局平均池化
        avg_score = F.adaptive_avg_pool2d(x, (1,1))
        avg_score = rearrange(avg_score, 'b c 1 1 -> b c')
        avg_score = self.avg_kan01(avg_score)
        avg_score = self.avg_kan02(avg_score)

        # 全局最大池化
        max_score = F.adaptive_max_pool2d(x, (1,1))
        max_score = rearrange(max_score, 'b c 1 1 -> b c')
        max_score = self.max_kan01(max_score)
        max_score = self.max_kan02(max_score)

        # 拼接 avg+max
        score = self.sum_kan(torch.cat([avg_score, max_score], dim=1))
        score = rearrange(score, 'b c -> b c 1 1')

        # 加权 + 残差
        x = x * score + x
        return x

# -----------------------------
# 新的 CID 模块：DConv7 + SpeKAN
# -----------------------------
class CID(nn.Module):
    def __init__(self, f_number):
        super().__init__()
        self.channel_independent = DConv7(f_number)
        self.channel_dependent = SpeKAN(f_number)  # 用 SpeKAN 替代 KAN/MLP

    def forward(self, x):
        x = self.channel_independent(x)
        x = self.channel_dependent(x)
        return x

# -----------------------------
# 测试
# -----------------------------
if __name__ == '__main__':
    input = torch.randn(1, 64, 128, 128)
    model = CID(64)
    output = model(input)
    print(f"输入张量形状: {input.shape}")
    print(f"输出张量形状: {output.shape}")

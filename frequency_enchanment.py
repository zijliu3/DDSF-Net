import torch
import torch.nn as nn


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):

        ctx.eps = eps

        N, C, H, W = x.size()

        mu = x.mean(1, keepdim=True)

        var = (x - mu).pow(2).mean(1, keepdim=True)

        y = (x - mu) / (var + eps).sqrt()

        ctx.save_for_backward(y, var, weight)

        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):

        eps = ctx.eps

        N, C, H, W = grad_output.size()

        y, var, weight = ctx.saved_variables

        g = grad_output * weight.view(1, C, 1, 1)

        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)

        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)

        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()

        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))

        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))

        self.eps = eps

    def forward(self, x):

        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class FreMLP(nn.Module):
    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()

        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):

        _, _, H, W = x.shape

        x_freq = torch.fft.rfft2(x, norm='backward')

        mag = torch.abs(x_freq)

        pha = torch.angle(x_freq)

        mag = self.process1(mag)

        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out

class Frequency_Domain(nn.Module):
    def __init__(self, channels,num_heads, bias=False):
        super().__init__()

        self.num_heads = num_heads
        self.norm = LayerNorm2d(channels)

        self.freq = FreMLP(nc=channels, expand=2)

        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))


        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=(1, 1, 1), bias=bias)

        self.qkv_dwconv = nn.Conv3d(
            channels * 3, channels * 3, kernel_size=(3, 3, 3),
            stride=1, padding=1, groups=channels* 3, bias=bias
        )

        self.project_out = nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), bias=bias)

        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)


        self.dep_conv = nn.Conv3d(
            9 * channels // self.num_heads, channels, kernel_size=(3, 3, 3),
            bias=True, groups=channels // self.num_heads, padding=1
        )

    def forward(self, inp):
        b, c, h, w = inp.shape

        x = inp.unsqueeze(2)


        qkv = self.qkv_dwconv(self.qkv(x))

        qkv = qkv.squeeze(2)


        f_conv = qkv.permute(0, 2, 3, 1)

        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)

        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)


        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)

        out_conv = self.dep_conv(f_conv)
        out_conv = out_conv.squeeze(2)
        ##################################################################################################################################################################

        A = inp

        x_step2 = self.norm(inp)
        x_freq = self.freq(x_step2)

        x = A * x_freq

        x = A + x * self.gamma
        x = x +out_conv
        return x


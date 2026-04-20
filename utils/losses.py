import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
import numpy as np
from typing import Tuple

# ---------------- Gaussian Kernel ----------------
def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2) ** 2 / float(2 * sigma ** 2)
    gauss = torch.stack([torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()

def get_gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
    if not isinstance(ksize, int) or ksize % 2 == 0 or ksize <= 0:
        raise TypeError(f"ksize must be an odd positive integer. Got {ksize}")
    return gaussian(ksize, sigma)

def get_gaussian_kernel2d(ksize: Tuple[int,int], sigma: Tuple[float,float]) -> torch.Tensor:
    kx, ky = ksize
    sx, sy = sigma
    kernel_x = get_gaussian_kernel(kx, sx)
    kernel_y = get_gaussian_kernel(ky, sy)
    kernel_2d = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d

# ---------------- PSNR Loss ----------------

# ---------------- SSIM Loss ----------------
class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, reduction: str = 'mean', max_val: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.max_val = max_val
        self.reduction = reduction
        self.window = get_gaussian_kernel2d((window_size, window_size), (1.5, 1.5))
        self.padding = (window_size - 1) // 2
        self.C1 = (0.01 * max_val) ** 2
        self.C2 = (0.03 * max_val) ** 2

    def filter2D(self, input, kernel, channel):
        return F.conv2d(input, kernel, padding=self.padding, groups=channel)

    def forward(self, img1, img2):
        b, c, h, w = img1.shape
        kernel = self.window.to(img1.device).to(img1.dtype).repeat(c, 1, 1, 1)
        mu1 = self.filter2D(img1, kernel, c)
        mu2 = self.filter2D(img2, kernel, c)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
        sigma1_sq = self.filter2D(img1*img1, kernel, c) - mu1_sq
        sigma2_sq = self.filter2D(img2*img2, kernel, c) - mu2_sq
        sigma12 = self.filter2D(img1*img2, kernel, c) - mu1_mu2
        ssim_map = ((2*mu1_mu2+self.C1)*(2*sigma12+self.C2)) / ((mu1_sq+mu2_sq+self.C1)*(sigma1_sq+sigma2_sq+self.C2))
        loss = torch.clamp(1. - ssim_map, min=0, max=1) / 2.
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

# ---------------- Charbonnier Loss ----------------
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff*diff + self.eps**2))
        return loss

# ---------------- Edge Loss ----------------
class GradientLoss(torch.nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def gradient(self, img):
        # Sobel horizontal
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]
        # Sobel vertical
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx, gy

    def forward(self, pred, target):
        px, py = self.gradient(pred)
        tx, ty = self.gradient(target)
        return (px - tx).abs().mean() + (py - ty).abs().mean()

# ---------------- VGG Perceptual Loss ----------------
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        # self.criterion = nn.L1Loss()
        self.criterion = nn.L1Loss(reduction='sum')
        self.criterion2 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward2(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion2(x_vgg[i], y_vgg[i].detach())
        return loss


# ---------------- Usage Example ----------------
# 设置各损失权重
lambda_ssim = 14.0
lambda_vgg = 0.012
lambda_edge = 1.5

# 定义损失函数
char_loss_fn = CharbonnierLoss()
ssim_loss_fn = SSIMLoss()
edge_loss_fn = GradientLoss()
vgg_loss_fn = VGGLoss()

# 在训练中计算总损失
# loss = (char_loss_fn(pred, target)
#         + lambda_ssim * ssim_loss_fn(pred, target)
#         + lambda_vgg * vgg_loss_fn(pred, target)
#         + lambda_edge * edge_loss_fn(pred, target))

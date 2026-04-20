import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
from wtconv.wtconv2d import *
from MultiScaleDirectional_Conv import *
from CID import *
from Wavelet_Inverse_Transform import *
class HeightWidthDiagonalFeatureProcessor(nn.Module):
    def __init__(self, input_channel_count, output_channel_count):
        super(HeightWidthDiagonalFeatureProcessor, self).__init__()


        self.discrete_wavelet_transform = DWTForward(
            J=1, mode='zero', wave='haar'
        )
        self.dconv_1 = nn.Conv2d(input_channel_count, output_channel_count, kernel_size=3, padding=1, padding_mode='reflect',groups=input_channel_count)
        self.dconv_2 = nn.Conv2d(input_channel_count, output_channel_count,kernel_size=3, padding=1, padding_mode='reflect',groups=input_channel_count)
        self.dconv_3 = nn.Conv2d(input_channel_count, output_channel_count, kernel_size=3, padding=1, padding_mode='reflect',groups=input_channel_count)
        self.wtconv =  WTConv2d(input_channel_count, output_channel_count, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1')
        self.conv_l = nn.Conv2d(input_channel_count, output_channel_count,kernel_size=3, padding=1, padding_mode='reflect',groups=input_channel_count)
        self.directionconv_1 =  nn.Conv2d(
            input_channel_count,output_channel_count,
            kernel_size=(1, 7),
            padding=(0, 3),
            stride=1,
            groups=input_channel_count
        )
        self.directionconv_2 =  nn.Conv2d(
            input_channel_count,output_channel_count,
            kernel_size=(7, 1),
            padding=(3, 0),
            stride=1,
            groups=input_channel_count
        )
        self.directionconv_3 = MultiScaleDirectionalConvBlock(input_channel_count)
        self.cid_h = CID(input_channel_count)
        self.cid_v = CID(input_channel_count)
        self.cid_d = CID(input_channel_count)
        self.wiTr =  WITR()
        ##################################################################################
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d( input_channel_count, output_channel_count // 16, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(input_channel_count //16,input_channel_count, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input_tensor):

        low_frequency_component, high_frequency_components = self.discrete_wavelet_transform(input_tensor)


        horizontal_detail_coefficient = high_frequency_components[0][:, :, 0, :, :]
        vertical_detail_coefficient   = high_frequency_components[0][:, :, 1, :, :]
        diagonal_detail_coefficient   = high_frequency_components[0][:, :, 2, :, :]

        low_frequency_component_new =  self.wtconv(low_frequency_component)
        low_frequency_component_new_1 = self.directionconv_1(low_frequency_component_new)
        low_frequency_component_new_2 = self.directionconv_2(low_frequency_component_new)
        low_frequency_component_new_3 = self.directionconv_3(low_frequency_component_new)
        horizontal_detail_coefficient = self.dconv_1(horizontal_detail_coefficient) + low_frequency_component_new_1
        vertical_detail_coefficient =  self.dconv_2(vertical_detail_coefficient) + low_frequency_component_new_2
        diagonal_detail_coefficient = self.dconv_3(diagonal_detail_coefficient) + low_frequency_component_new_3

        high_features = torch.cat(
            [
                horizontal_detail_coefficient,
                vertical_detail_coefficient,
                diagonal_detail_coefficient
            ],
            dim=1
        )
        x =  self.wiTr(low_frequency_component_new, high_features)

        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        scale = avg_out + max_out
        x = input_tensor* scale
        return x




if __name__ == '__main__':
    model = HeightWidthDiagonalFeatureProcessor(16, 16)
    input_tensor = torch.rand(1, 16, 55, 150)

    x = model(input_tensor)

    print(f'Input size: {input_tensor.size()}')
    print(f'x size: {x.size()}')


    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")

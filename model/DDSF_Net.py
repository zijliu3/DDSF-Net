
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

import einops
from einops import rearrange
import numpy as np
from frequency_fusion import *
from frequency_enchanment import *
from MSA import *
##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c, embed_dim, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x
##########################################################################
## Resizing modules
class Downsample(nn.Module):#尺寸减半，通道数乘以2
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
##########################################################################

class DDSF_Net(nn.Module):
    def __init__(self,

        dim = 16,
        num_blocks = [1,1,2,2],
        inp_channels=3,
        out_channels=3,
        num_refinement_blocks = 2,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',
        attention=True,
        skip = False
    ):

        super(DDSF_Net, self).__init__()

        self.coefficient = nn.Parameter(torch.Tensor(np.ones((4, 2,int(int(dim * 2 * 4))))),
                                        requires_grad=attention)

        self.patch_embed = OverlapPatchEmbed(3, 16)
        self.patch_embed_final = OverlapPatchEmbed(16, 3)
        self.down_1 = Downsample(16)
        self.down_2 = Downsample(32)
        self.down_3 = Downsample(64)
        #self.down_4 = Downsample(128)

        #self.up_1 = Upsample(256)
        self.up_2 = Upsample(128)
        self.up_3 = Upsample(64)
        self.up_4 = Upsample(32)
        #######################################################################################################################################
        self.hfeb_1 = nn.Sequential(*[HeightWidthDiagonalFeatureProcessor(16, 16) for _ in range(num_blocks[0])])
        self.hfeb_2 = nn.Sequential(*[HeightWidthDiagonalFeatureProcessor(32, 32) for _ in range(num_blocks[1])])
        self.hfeb_3 = nn.Sequential(*[HeightWidthDiagonalFeatureProcessor(64, 64) for _ in range(num_blocks[2])])
        self.hfeb_4 = nn.Sequential(*[HeightWidthDiagonalFeatureProcessor(128, 128) for _ in range(num_blocks[2])])
        self.hfeb_6 = nn.Sequential(*[HeightWidthDiagonalFeatureProcessor(128, 128) for _ in range(num_blocks[2])])
        self.hfeb_7 = nn.Sequential(*[HeightWidthDiagonalFeatureProcessor(64, 64) for _ in range(num_blocks[2])])
        self.hfeb_8 = nn.Sequential(*[HeightWidthDiagonalFeatureProcessor(32, 32) for _ in range(num_blocks[1])])
        self.hfeb_9 = nn.Sequential(*[HeightWidthDiagonalFeatureProcessor(16, 16) for _ in range(num_blocks[0])])
 ###############################################################################################################################################
        self.encoder_1_L = nn.Sequential(*[Frequency_Domain(16, 1) for _ in range(num_blocks[0])])
        self.encoder_2_L = nn.Sequential(*[Frequency_Domain(32, 2) for _ in range(num_blocks[1])])
        self.encoder_3_L = nn.Sequential(*[Frequency_Domain(64, 4) for _ in range(num_blocks[2])])
        self.encoder_4_L = nn.Sequential(*[Frequency_Domain(128, 8) for _ in range(num_blocks[3])])
        self.encoder_6_L = nn.Sequential(*[Frequency_Domain(128, 8) for _ in range(num_blocks[3])])
        self.encoder_7_L = nn.Sequential(*[Frequency_Domain(64, 4) for _ in range(num_blocks[2])])
        self.encoder_8_L = nn.Sequential(*[Frequency_Domain(32, 2) for _ in range(num_blocks[1])])
        self.encoder_9_L = nn.Sequential(*[Frequency_Domain(16, 1) for _ in range(num_blocks[0])])
 ##############################################################################################################################################################

        ### skip connection wit weights
        self.coefficient_4_6 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim * 2 * 4))))), requires_grad=attention)
        self.coefficient_3_7 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim * 2 * 2))))), requires_grad=attention)
        self.coefficient_2_8 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim * 2))))), requires_grad=attention)
        self.coefficient_1_9 = nn.Parameter(torch.Tensor(np.ones((2, int(int(dim))))), requires_grad=attention)

        ### skip then conv 1x1
        self.skip_4_6 = nn.Conv2d(int(int(dim * 2 * 4)), int(int(dim * 2 * 4)), kernel_size=1, bias=bias)
        self.skip_3_7 = nn.Conv2d(int(int(dim * 2 * 2)), int(int(dim * 2 * 2)), kernel_size=1, bias=bias)
        self.skip_2_8 = nn.Conv2d(int(int(dim * 2)), int(int(dim * 2)), kernel_size=1, bias=bias)
        self.skip_1_9 = nn.Conv2d(int(int(dim )), int(int(dim )), kernel_size=1, bias=bias)
################################################################################################################################################################
        self.msa1_9 =   DPCF(16,16)
        self.msa2_8 = DPCF(32, 32)
        self.msa3_7 =  DPCF(64, 64)

    def forward(self, inp_img):

        h, w = inp_img.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        inp_img = F.pad(inp_img, (0, pad_w, 0, pad_h), mode='reflect')
###################################################################################################################################################################
        inp_enc_encoder1 = self.patch_embed(inp_img)#3——16
        inp_enc_encoder1 = self.hfeb_1(inp_enc_encoder1)#c16
        inp_enc_encoder1 = self.encoder_1_L(inp_enc_encoder1)

        inp_enc_encoder2 = self.down_1(inp_enc_encoder1)#16-32

        inp_enc_encoder2 = self.hfeb_2(inp_enc_encoder2)
        inp_enc_encoder2 = self.encoder_2_L(inp_enc_encoder2)

        inp_enc_encoder3 = self.down_2(inp_enc_encoder2)#32-64

        inp_enc_encoder3 = self.hfeb_3(inp_enc_encoder3)
        inp_enc_encoder3 = self.encoder_3_L(inp_enc_encoder3)

        inp_enc_encoder4 = self.down_3(inp_enc_encoder3)  # 64-128


        inp_enc_encoder4 = self.hfeb_4(inp_enc_encoder4)
        inp_enc_encoder4 = self.encoder_4_L(inp_enc_encoder4)


        inp_enc_encoder6 = self.hfeb_6(inp_enc_encoder4)

        inp_enc_encoder6 = self.encoder_6_L(inp_enc_encoder4)

        inp_enc_encoder7 = self.up_2(inp_enc_encoder6)  # 128-64

        inp_enc_encoder3_7 =self.msa3_7(inp_enc_encoder3,inp_enc_encoder7)


        inp_enc_encoder7 = self.hfeb_7( inp_enc_encoder3_7)
        inp_enc_encoder7 = self.encoder_7_L(inp_enc_encoder7)

        inp_enc_encoder8 = self.up_3(inp_enc_encoder7)  # 64-32

        inp_enc_encoder2_8 =self.msa2_8(inp_enc_encoder2,inp_enc_encoder8)


        inp_enc_encoder8 = self.hfeb_8(inp_enc_encoder2_8)
        inp_enc_encoder8 = self.encoder_8_L(inp_enc_encoder8)

        inp_enc_encoder9 = self.up_4(inp_enc_encoder8)  # 32-16

        inp_enc_encoder1_9 = self.msa1_9(inp_enc_encoder1,inp_enc_encoder9)

        inp_enc_encoder9 = self.hfeb_9( inp_enc_encoder1_9)
        inp_enc_encoder9 = self.encoder_9_L(inp_enc_encoder9)

        out = self.patch_embed_final(inp_enc_encoder9)
        out = out + inp_img
        out = out[:, :, :h, :w]
        return out


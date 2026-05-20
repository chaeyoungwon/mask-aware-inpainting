import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_bn=True,
                 activation=nn.ReLU(inplace=True)):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if activation is not None:
            layers.append(activation)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_ch, out_ch, activation=nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x, skip):
        # upsample 먼저, 그 다음 skip과 concat
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class CAE(nn.Module):
    """
    Vanilla Convolutional AutoEncoder (U-Net style) for inpainting.
    Input:  masked_image (B, 3, H, W)
    Output: reconstructed image (B, 3, H, W)
    """

    def __init__(self, base_ch=64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(3,         base_ch,    use_bn=False)  # 256
        self.enc2 = ConvBlock(base_ch,   base_ch*2,  stride=2)      # 128
        self.enc3 = ConvBlock(base_ch*2, base_ch*4,  stride=2)      # 64
        self.enc4 = ConvBlock(base_ch*4, base_ch*8,  stride=2)      # 32
        self.enc5 = ConvBlock(base_ch*8, base_ch*8,  stride=2)      # 16

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch*8, base_ch*8, stride=2) # 8

        # Decoder (skip connections)
        self.dec5 = DeconvBlock(base_ch*8 + base_ch*8, base_ch*8)   # 16
        self.dec4 = DeconvBlock(base_ch*8 + base_ch*8, base_ch*8)   # 32
        self.dec3 = DeconvBlock(base_ch*8 + base_ch*4, base_ch*4)   # 64
        self.dec2 = DeconvBlock(base_ch*4 + base_ch*2, base_ch*2)   # 128
        self.dec1 = DeconvBlock(base_ch*2 + base_ch,   base_ch)     # 256

        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x, mask=None):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        bt = self.bottleneck(e5)

        d = self.dec5(bt, e5)
        d = self.dec4(d,  e4)
        d = self.dec3(d,  e3)
        d = self.dec2(d,  e2)
        d = self.dec1(d,  e1)

        return self.tanh(self.out_conv(d))

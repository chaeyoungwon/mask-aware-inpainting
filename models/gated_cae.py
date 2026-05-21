import torch
import torch.nn as nn

from models.gated_conv_layer import GatedConv2d


class GatedConvBlock(nn.Module):
    """Encoder block: GatedConv2d with optional stride-2 downsampling."""

    def __init__(self, in_ch, out_ch, stride=1, use_bn=True):
        super().__init__()
        self.gconv = GatedConv2d(in_ch, out_ch, stride=stride, use_bn=use_bn)

    def forward(self, x):
        return self.gconv(x)


class GatedDeconvBlock(nn.Module):
    """Decoder block: bilinear upsample → concat skip → GatedConv2d."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.gconv = GatedConv2d(in_ch, out_ch,
                                 activation=nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x, skip):
        return self.gconv(torch.cat([self.up(x), skip], dim=1))


class GatedCAE(nn.Module):
    """
    Gated Convolutional AutoEncoder (U-Net style) for inpainting.

    Input : masked_image (B, 3, H, W)  +  mask (B, 1, H, W)
    Output: reconstructed image (B, 3, H, W)

    The first layer receives [masked_image ‖ mask] (4 channels), so the
    network has explicit knowledge of hole boundaries from the start.
    Subsequent layers use learnable soft gating instead of the rule-based
    binary mask update used in PConv, giving per-channel, per-location
    flexibility.
    """

    def __init__(self, base_ch=64):
        super().__init__()

        # Encoder  (spatial: 256 → 128 → 64 → 32 → 16 → 8)
        self.enc1      = GatedConvBlock(4,          base_ch,   use_bn=False)  # 4 = 3 RGB + 1 mask
        self.enc2      = GatedConvBlock(base_ch,    base_ch*2, stride=2)
        self.enc3      = GatedConvBlock(base_ch*2,  base_ch*4, stride=2)
        self.enc4      = GatedConvBlock(base_ch*4,  base_ch*8, stride=2)
        self.enc5      = GatedConvBlock(base_ch*8,  base_ch*8, stride=2)
        self.bottleneck= GatedConvBlock(base_ch*8,  base_ch*8, stride=2)

        # Decoder with skip connections  (8 → 16 → 32 → 64 → 128 → 256)
        self.dec5 = GatedDeconvBlock(base_ch*8 + base_ch*8, base_ch*8)
        self.dec4 = GatedDeconvBlock(base_ch*8 + base_ch*8, base_ch*8)
        self.dec3 = GatedDeconvBlock(base_ch*8 + base_ch*4, base_ch*4)
        self.dec2 = GatedDeconvBlock(base_ch*4 + base_ch*2, base_ch*2)
        self.dec1 = GatedDeconvBlock(base_ch*2 + base_ch,   base_ch)

        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=1)
        self.tanh     = nn.Tanh()

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device)

        inp = torch.cat([x, mask], dim=1)   # (B, 4, H, W)

        e1 = self.enc1(inp)
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

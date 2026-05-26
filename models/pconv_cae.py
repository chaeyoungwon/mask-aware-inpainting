import torch
import torch.nn as nn

from models.pconv_layer import PartialConv2d


class PConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_bn=True,
                 activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.pconv = PartialConv2d(in_ch, out_ch, stride=stride)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.activation = activation

    def forward(self, x, mask):
        # 부분 컨볼루션을 수행하고 업데이트된 마스크를 함께 전달
        x, mask = self.pconv(x, mask)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x, mask


class PConvDeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_mask = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = PConvBlock(in_ch, out_ch,
                               activation=nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x, mask, skip, skip_mask):
        # feature와 mask를 각각 업샘플링
        x = self.up(x)
        mask = self.up_mask(mask)
        # 인코더 경로의 skip feature를 concat
        feat = torch.cat([x, skip], dim=1)
        # 디코더와 skip 경로의 마스크를 결합
        combined_mask = torch.clamp(mask + skip_mask, 0, 1)
        return self.conv(feat, combined_mask)


class PConvCAE(nn.Module):
    """Partial Conv CAE with mask update for inpainting."""

    def __init__(self, base_ch=64):
        super().__init__()

        self.enc1 = PConvBlock(3, base_ch, use_bn=False)
        self.enc2 = PConvBlock(base_ch, base_ch * 2, stride=2)
        self.enc3 = PConvBlock(base_ch * 2, base_ch * 4, stride=2)
        self.enc4 = PConvBlock(base_ch * 4, base_ch * 8, stride=2)
        self.enc5 = PConvBlock(base_ch * 8, base_ch * 8, stride=2)
        self.bottleneck = PConvBlock(base_ch * 8, base_ch * 8, stride=2)

        self.dec5 = PConvDeconvBlock(base_ch * 8 + base_ch * 8, base_ch * 8)
        self.dec4 = PConvDeconvBlock(base_ch * 8 + base_ch * 8, base_ch * 8)
        self.dec3 = PConvDeconvBlock(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.dec2 = PConvDeconvBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec1 = PConvDeconvBlock(base_ch * 2 + base_ch, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device)

        # 인코더 경로: feature와 mask를 함께 전파
        e1, m1 = self.enc1(x, mask)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)
        e4, m4 = self.enc4(e3, m3)
        e5, m5 = self.enc5(e4, m4)
        bt, m_bt = self.bottleneck(e5, m5)

        # 디코더 경로: 업샘플, skip 연결, 부분 컨볼루션 계속 수행
        d5, m_d5 = self.dec5(bt, m_bt, e5, m5)
        d4, m_d4 = self.dec4(d5, m_d5, e4, m4)
        d3, m_d3 = self.dec3(d4, m_d4, e3, m3)
        d2, m_d2 = self.dec2(d3, m_d3, e2, m2)
        d1, m_d1 = self.dec1(d2, m_d2, e1, m1)

        return self.tanh(self.out_conv(d1))
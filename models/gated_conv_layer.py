import torch.nn as nn


class GatedConv2d(nn.Module):
    """
    Gated Convolution layer (Yu et al., 2019).

    Computes two parallel convolutions on the same input:
      Feature_{y,x} = phi( BN( W_f * x ) )   <- content branch
      Gating_{y,x}  = sigma( W_g * x )        <- soft mask branch
      Output        = Feature  *  Gating       <- element-wise

    phi   : configurable activation (default ELU)
    sigma : sigmoid, squashes gating to [0, 1]

    Unlike PConv, the gating values are learnable and continuous per
    channel and spatial location, so the network decides how much to
    trust each feature rather than following a fixed binary rule.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, use_bn=True, activation=None):
        super().__init__()
        # bias is redundant when BN follows, so omit it there
        self.feature_conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, padding, dilation,
            bias=not use_bn,
        )
        # gating branch always has its own bias (no BN here)
        self.gating_conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, padding, dilation,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.activation = activation if activation is not None else nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = self.activation(self.bn(self.feature_conv(x)))
        gating  = self.sigmoid(self.gating_conv(x))
        return feature * gating

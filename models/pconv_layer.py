import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Module):
    """Partial convolution layer from Liu et al. 2018.

    The forward pass computes a convolution over known pixels only, then
    normalizes by the valid pixel count. It also returns an updated binary
    mask for the next layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            bias=bias,
        )

        # 마스크 내 유효 픽셀을 세기 위해 모두 1인 커널을 사용
        self.register_buffer(
            'weight_mask_updater',
            torch.ones(1, 1, kernel_size, kernel_size),
        )
        self.kernel_area = self.weight_mask_updater.sum().item()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, input, mask=None):
        if mask is None:
            mask = torch.ones(
                input.size(0), 1, input.size(2), input.size(3),
                device=input.device,
                dtype=input.dtype,
            )

        with torch.no_grad():
            # 각 컨볼루션 윈도우에서 유효 픽셀 수를 계산
            updated_mask = F.conv2d(
                mask, self.weight_mask_updater,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation,
            )
            # 유효 픽셀만 존재할 때 출력을 보정하기 위한 비율
            mask_ratio = self.kernel_area / (updated_mask + 1e-8)
            updated_mask = torch.where(updated_mask > 0, torch.ones_like(updated_mask), torch.zeros_like(updated_mask))
            mask_ratio = mask_ratio * updated_mask

        # 유효한 픽셀 영역에 대해서만 컨볼루션을 적용하고, 출력 보정
        output = self.input_conv(input * mask)

        if self.input_conv.bias is not None:
            bias_view = self.input_conv.bias.view(1, -1, 1, 1)
            output = (output - bias_view) * mask_ratio + bias_view
        else:
            output = output * mask_ratio

        output = output * updated_mask
        return output, updated_mask
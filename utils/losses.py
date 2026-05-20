import ssl
import torch.nn as nn
import torchvision.models as models

ssl._create_default_https_context = ssl._create_unverified_context

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # pool1, pool2, pool3 레이어까지 슬라이스
        self.slice1 = nn.Sequential(*list(vgg.features)[:5])   # pool1
        self.slice2 = nn.Sequential(*list(vgg.features)[5:10]) # pool2
        self.slice3 = nn.Sequential(*list(vgg.features)[10:17])# pool3
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        return f1, f2, f3


class InpaintingLoss(nn.Module):
    def __init__(self, perceptual_weight=0.05):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.vgg = VGGPerceptualLoss()
        self.perceptual_weight = perceptual_weight

    def forward(self, output, gt, mask):
        """
        output: 모델 출력 (B, 3, H, W)
        gt:     ground truth (B, 3, H, W)
        mask:   (B, 1, H, W), hole=0 non-hole=1
        """
        # hole/non-hole 분리
        l_valid = self.l1(output * mask,       gt * mask)
        l_hole  = self.l1(output * (1 - mask), gt * (1 - mask))

        # I_comp: hole 부분만 output, 나머지는 gt
        i_comp = output * (1 - mask) + gt * mask

        # perceptual loss
        l_perc = 0
        for feat_out, feat_comp, feat_gt in zip(
            self.vgg(output), self.vgg(i_comp), self.vgg(gt)
        ):
            n = feat_gt.numel()
            l_perc += (feat_out - feat_gt).abs().sum() / n
            l_perc += (feat_comp - feat_gt).abs().sum() / n

        total = l_valid + 6.0 * l_hole + self.perceptual_weight * l_perc
        return total, {'l_valid': l_valid, 'l_hole': l_hole, 'l_perc': l_perc}

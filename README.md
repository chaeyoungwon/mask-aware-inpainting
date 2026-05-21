# Mask-Aware Inpainting

Image inpainting 기법을 단계적으로 구현하고 비교하는 프로젝트입니다.

- **CAE** (Vanilla Convolutional AutoEncoder) → baseline
- **PConv CAE** (Partial Convolution) → Liu et al., 2018 아이디어를 CAE 구조에 적용
- **Gated Conv CAE** (Gated Convolution) → Yu et al., 2019

## 프로젝트 구조

```
mask-aware-inpainting/
├── config.py                   # 하이퍼파라미터
├── train.py                    # 학습
├── evaluate.py                 # 평가 (PSNR, SSIM)
├── datasets/
│   ├── celeba_dataset.py       # CelebA 로더
│   └── mask_generator.py       # 랜덤 stroke mask 생성
├── models/
│   ├── cae.py                  # Vanilla CAE (baseline)
│   ├── pconv_layer.py          # Partial Conv layer
│   ├── pconv_cae.py            # PConv 기반 CAE
│   ├── gated_conv_layer.py     # Gated Conv layer
│   └── gated_cae.py            # Gated Conv 기반 CAE
└── utils/
    ├── losses.py               # L1 + Perceptual loss
    ├── metrics.py              # PSNR, SSIM
    └── visualization.py        # 이미지 저장
```

## 환경 설정

```bash
pip install torch torchvision
```

## 데이터

CelebA 데이터셋을 사용합니다. 첫 실행 시 자동 다운로드됩니다.

```
data/
└── celeba/
    ├── img_align_celeba/
    ├── list_attr_celeba.txt
    └── ...
```

기본 경로는 `./data`이며 `config.py`에서 변경 가능합니다.

## 학습

```bash
python3 train.py
```

`config.py`에서 주요 설정을 변경할 수 있습니다.

| 파라미터     | 기본값 | 설명             |
| ------------ | ------ | ---------------- |
| `image_size` | 256    | 입력 이미지 크기 |
| `batch_size` | 16     | 배치 크기        |
| `num_epochs` | 100    | 학습 에폭 수     |
| `lr`         | 2e-4   | 학습률           |

## 평가

```bash
python3 evaluate.py checkpoints/epoch010.pth
```

## Loss

| 항목           | 가중치 | 설명             |
| -------------- | ------ | ---------------- |
| `L_valid`      | 1.0    | non-hole 픽셀 L1 |
| `L_hole`       | 6.0    | hole 픽셀 L1     |
| `L_perceptual` | 0.05   | VGG16 feature L1 |

## 참고 논문

- [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723) (Liu et al., 2018)
- [Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589) (Yu et al., 2019)

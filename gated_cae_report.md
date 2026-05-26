# Gated CAE 구현 및 실험 결과 보고

---

## 1. Gated Convolution 도입 배경

### Vanilla Convolution의 한계

이미지 복원(Inpainting) 태스크에서 Vanilla Convolution은 마스크 영역(손상된 픽셀)과 유효 영역을 구분하지 않고 동일하게 처리한다. 이로 인해 마스크 경계에서 색상 불일치, 흐릿함(blur), 아티팩트(artifact)가 발생하며 복원 품질이 저하된다.

### Partial Convolution의 한계

이를 개선하기 위해 제안된 Partial Convolution은 이진 마스크(Binary Mask)로 유효/무효 픽셀을 구분하지만, 다음의 한계가 있다.

- 마스크가 이진값(0 또는 1)으로만 표현되어 섬세한 경계 처리가 어려움
- 규칙 기반의 하드 게이팅으로 마스크 정보가 깊은 레이어에서 희석됨
- 모든 채널이 동일한 마스크를 공유하여 유연성이 낮음

### Gated Convolution의 해결 방식

Gated Convolution은 **학습 가능한 소프트 게이팅(Soft Gating)** 을 도입하여 위 문제를 해결한다. 이진 마스크 대신, 데이터로부터 자동으로 학습된 연속적인 게이팅 값(0~1)이 각 위치와 채널에서 특징의 중요도를 동적으로 조절한다.

$$y = \varphi(\text{Feature}) \odot \sigma(\text{Gating})$$

- **Feature**: 컨볼루션으로 계산된 특징 맵
- **Gating**: 별도의 컨볼루션 + Sigmoid를 통해 학습된 0~1 사이의 제어 신호
- **⊙**: 요소별 곱셈 — Gating 값이 Feature를 채널/위치별로 조절

---

## 2. Vanilla / Partial Conv / Gated Conv 비교

| 항목 | Vanilla Conv | Partial Conv | **Gated Conv** |
|:---|:---|:---|:---|
| 마스크 처리 | 없음 | 이진 마스크 | 학습된 소프트 게이팅 |
| 게이팅 방식 | - | 하드 (규칙 기반) | 소프트 (데이터 기반) |
| 채널별 처리 | 동일 | 동일 마스크 공유 | 채널/위치별 독립 |
| 경계 아티팩트 | 많음 | 감소 | 최소화 |
| 적응성 | 낮음 | 낮음 | 높음 |

---

## 3. 구현: Gated CAE (Gated Convolutional AutoEncoder)

### 아키텍처 개요

Vanilla CAE, Partial Conv CAE와 동일한 Encoder-Decoder 구조를 유지하되, 모든 Convolution Layer를 Gated Convolution Layer로 교체하였다. 이를 통해 Conv 종류별 성능 차이를 공정하게 비교할 수 있도록 설계하였다.

- **입력**: 마스크된 이미지 (128×128, RGB)
- **Encoder**: Gated Conv 기반 다운샘플링
- **Decoder**: Transposed Conv 기반 업샘플링
- **출력**: 복원된 이미지 (128×128, RGB)

### 학습 설정

| 항목 | 설정값 |
|:---|:---|
| 데이터셋 | CelebA (학습 5,000장 / 테스트) |
| 이미지 크기 | 128×128 |
| 배치 크기 | 16 |
| Learning Rate | 2e-4 |
| Loss | Valid Loss (×1.0) + Hole Loss (×6.0) + Perceptual Loss (×0.05) |
| 에폭 | 20 |

---

## 4. 실험 결과

### 4-1. 정량적 평가 (PSNR / SSIM)

| Epoch | PSNR (dB) | SSIM |
|:---:|:---:|:---:|
| 1 | 10.90 | 0.6046 |
| 5 | 13.37 | 0.6974 |
| 10 | 14.64 | 0.7425 |
| 15 | 15.48 | 0.7677 |
| **20** | **15.73** | **0.7635** |

**최종 성능: PSNR 15.73 dB / SSIM 0.7635**

### 4-2. 학습 곡선 분석

**수렴 특성**
- Epoch 1→6 구간에서 PSNR이 10.9 → 14.5 dB로 빠르게 상승하며 초기 학습이 효과적으로 진행됨
- Epoch 7 이후 완만하게 수렴하는 전형적인 학습 패턴을 보임
- Epoch 20 시점에도 완전한 plateau에 도달하지 않았으므로, 추가 학습 시 성능 향상 여지가 있음

**진동(Oscillation)**
- Epoch 5, 10, 13에서 일시적인 성능 하락이 관찰됨
- Learning Rate Scheduler(예: Cosine Annealing) 적용 시 개선 가능

### 4-3. 정성적 평가 (시각화)

아래 결과는 테스트셋에서 샘플링한 4장의 이미지에 대해 마스크 입력, 복원 결과, 원본을 비교한 것이다.

```
1행: Masked Input  (마스크된 입력)
2행: GatedCAE Output  (복원 결과)
3행: Ground Truth  (원본)
```

**관찰 결과**
- 얼굴 면적의 30~50%가 마스크된 상황에서도 얼굴 구조(눈, 코, 입 위치)가 자연스럽게 복원됨
- 단색·밝은 배경의 이미지(3, 4번 열)는 원본과 거의 구분되지 않는 수준으로 복원
- 배경이 복잡하거나 조명이 어두운 이미지(1, 2번 열)에서는 복원 이미지가 다소 blurry하게 나타남 — Pixel-level loss 최소화에 집중하는 구조적 특성으로 분석됨

---

## 5. 결론

Gated Convolution을 적용한 AutoEncoder는 마스크 영역을 학습 기반의 소프트 게이팅으로 처리함으로써, 마스크 경계 아티팩트를 줄이고 자연스러운 복원 결과를 생성하였다. 20 에폭 학습 기준으로 PSNR 15.73 dB, SSIM 0.7635를 달성하였으며, 시각적으로도 얼굴 구조를 안정적으로 복원하는 것을 확인하였다.

향후 Vanilla CAE, Partial Conv CAE와의 정량/정성 비교를 통해 Gated Convolution의 실질적인 성능 우위를 검증할 예정이다.

import numpy as np
import torch
from PIL import Image, ImageDraw
import random
import math


def generate_stroke_mask(height=256, width=256, max_strokes=20, max_len=100, max_width=20):
    """
    Random brush stroke mask.
    hole=0, non-hole=1 (논문 convention)
    """
    mask = np.ones((height, width), dtype=np.float32)
    num_strokes = random.randint(1, max_strokes)

    for _ in range(num_strokes):
        x = random.randint(0, width)
        y = random.randint(0, height)
        angle = random.uniform(0, 2 * math.pi)
        length = random.randint(10, max_len)
        brush_w = random.randint(5, max_width)
        num_vertex = random.randint(5, 20)

        img = Image.fromarray((mask * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)

        points = [(x, y)]
        for _ in range(num_vertex):
            angle += random.uniform(-math.pi / 4, math.pi / 4)
            step = random.randint(5, length)
            x = np.clip(x + int(step * math.cos(angle)), 0, width)
            y = np.clip(y + int(step * math.sin(angle)), 0, height)
            points.append((x, y))

        draw.line(points, fill=0, width=brush_w)
        for px, py in points:
            draw.ellipse(
                [px - brush_w // 2, py - brush_w // 2,
                 px + brush_w // 2, py + brush_w // 2],
                fill=0,
            )

        mask = np.array(img).astype(np.float32) / 255.0

    # random flip/rotate augmentation
    if random.random() > 0.5:
        mask = np.fliplr(mask)
    if random.random() > 0.5:
        mask = np.flipud(mask)
    k = random.randint(0, 3)
    mask = np.rot90(mask, k)

    return torch.from_numpy(mask.copy()).unsqueeze(0)  # (1, H, W)


def get_hole_ratio(mask):
    """마스크에서 hole(=0) 비율 반환"""
    return 1.0 - mask.mean().item()

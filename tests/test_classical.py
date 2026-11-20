import numpy as np
import pytest

from classical.pipeline import ClassicalConfig, detect, iou


def _synthetic(seed: int = 0):
    rng = np.random.default_rng(seed)
    img = np.full((200, 200), 128, np.uint8)
    img[80:110, 60:100] = 40  # dark "defect" at x 60-100, y 80-110
    noise = rng.normal(0, 2, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def test_iou_identical_and_disjoint():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_detects_synthetic_defect():
    boxes, mask = detect(_synthetic(), ClassicalConfig())
    assert mask.shape == (200, 200)
    assert any(iou(b, (60, 80, 100, 110)) >= 0.3 for b in boxes)


def test_rejects_color_input():
    with pytest.raises(ValueError):
        detect(np.zeros((10, 10, 3), np.uint8))
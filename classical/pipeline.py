"""Classical (non-deep-learning) defect detection baseline for NEU-DET.

Idea: steel defects are local deviations from the smooth background. We
remove the background, threshold the deviation, clean the mask with
morphology, and turn the remaining blobs into bounding boxes.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass(frozen=True)
class ClassicalConfig:
    blur_ksize: int = 5         # noise smoothing (odd)
    bg_ksize: int = 41          # background estimate, larger than any defect (odd)
    morph_ksize: int = 15        # structuring element for close/open
    min_area: int = 60          # px, drops speckle
    max_area_frac: float = 0.9  # drops "whole image" blobs


def preprocess(gray: np.ndarray, cfg: ClassicalConfig) -> np.ndarray:
    """Return |smoothed - background|, stretched to 0..255."""
    blurred = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0)
    background = cv2.GaussianBlur(gray, (cfg.bg_ksize, cfg.bg_ksize), 0)
    diff = cv2.absdiff(blurred, background)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)


def detect(gray: np.ndarray, cfg: ClassicalConfig = ClassicalConfig()) -> tuple[list[Box], np.ndarray]:
    """Detect defect candidates. Returns (boxes, binary mask)."""
    if gray.ndim != 2:
        raise ValueError("detect() expects a single-channel image")
    diff = preprocess(gray, cfg)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_ksize, cfg.morph_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = cfg.max_area_frac * gray.size
    boxes: list[Box] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg.min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, x + w, y + h))
    return boxes, mask


def class_from_name(path: Path) -> str:
    """rolled-in_scale_10.jpg -> rolled-in_scale."""
    return path.stem.rsplit("_", 1)[0]


def load_gt_boxes(xml_path: Path) -> list[Box]:
    """Read Pascal-VOC boxes from an annotation file."""
    root = ET.parse(xml_path).getroot()
    boxes: list[Box] = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        v = [int(float(bb.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax")]
        boxes.append((v[0], v[1], v[2], v[3]))
    return boxes


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
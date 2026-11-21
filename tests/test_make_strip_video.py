import cv2
import numpy as np

from ingest.make_strip_video import TILE, build_strip, write_camera


def _fake_dataset(tmp_path, n=3):
    img_dir, ann_dir = tmp_path / "imgs", tmp_path / "anns"
    img_dir.mkdir()
    ann_dir.mkdir()
    for i in range(n):
        img = np.full((TILE, TILE), 100 + i * 20, np.uint8)
        cv2.imwrite(str(img_dir / f"scratches_{i}.jpg"), img)
        (ann_dir / f"scratches_{i}.xml").write_text(
            "<annotation><object><bndbox><xmin>10</xmin><ymin>20</ymin>"
            "<xmax>60</xmax><ymax>50</ymax></bndbox></object></annotation>")
    return sorted(img_dir.glob("*.jpg")), ann_dir


def test_build_strip_offsets_defect_boxes_by_tile(tmp_path):
    paths, ann_dir = _fake_dataset(tmp_path)
    strip, defects = build_strip(paths, ann_dir, 4, np.random.default_rng(0))
    assert strip.shape == (4 * TILE, TILE)
    assert len(defects) == 4
    assert [d["box"][1] for d in defects] == [20, 220, 420, 620]  # y1 shifted by tile index
    assert all(d["cls"] == "scratches" for d in defects)


def test_write_camera_produces_expected_frame_count(tmp_path):
    paths, ann_dir = _fake_dataset(tmp_path)
    strip, _ = build_strip(paths, ann_dir, 4, np.random.default_rng(0))
    out = tmp_path / "cam.mp4"
    n = write_camera(out, strip, fps=10, seconds=2, speed=4)
    assert n == 20
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (TILE, TILE)
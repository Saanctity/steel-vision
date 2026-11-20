import cv2
import numpy as np

from detect import prepare_data as pd_


def test_convert_split_writes_normalised_yolo_labels(tmp_path, monkeypatch):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "train_images").mkdir(parents=True)
    (src / "train_annotations").mkdir(parents=True)
    cv2.imwrite(str(src / "train_images" / "scratches_1.jpg"), np.zeros((200, 200), np.uint8))
    (src / "train_annotations" / "scratches_1.xml").write_text(
        "<annotation><object><bndbox><xmin>20</xmin><ymin>40</ymin>"
        "<xmax>120</xmax><ymax>80</ymax></bndbox></object></annotation>")
    monkeypatch.setattr(pd_, "SRC", src)
    monkeypatch.setattr(pd_, "DST", dst)

    n_img, n_box = pd_.convert_split("train", "train")

    assert (n_img, n_box) == (1, 1)
    parts = (dst / "labels/train/scratches_1.txt").read_text().split()
    assert parts[0] == str(pd_.CLASSES.index("scratches"))
    got = [float(v) for v in parts[1:]]
    assert got == [0.35, 0.3, 0.5, 0.2]  # cx, cy, w, h
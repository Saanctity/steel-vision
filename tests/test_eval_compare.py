from detect.eval_compare import match_counts, prf


def test_perfect_match():
    gt = [(0, 0, 10, 10), (50, 50, 80, 80)]
    assert match_counts(gt, list(gt), 0.5) == (2, 0, 0)


def test_one_prediction_cannot_match_two_gt_boxes():
    gt = [(0, 0, 10, 10), (0, 0, 10, 10)]
    pred = [(0, 0, 10, 10)]
    assert match_counts(gt, pred, 0.5) == (1, 0, 1)


def test_extra_predictions_are_false_positives():
    gt = [(0, 0, 10, 10)]
    pred = [(0, 0, 10, 10), (100, 100, 120, 120), (150, 150, 170, 170)]
    assert match_counts(gt, pred, 0.5) == (1, 2, 0)


def test_low_overlap_below_threshold_is_miss():
    gt = [(0, 0, 10, 10)]
    pred = [(8, 8, 18, 18)]
    assert match_counts(gt, pred, 0.3) == (0, 1, 1)


def test_prf_values():
    pr, rc, f1 = prf(6, 2, 4)
    assert (round(pr, 2), round(rc, 2), round(f1, 3)) == (0.75, 0.6, 0.667)
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
from detect.benchmark import summarize


def test_summarize_median_and_p95():
    med, p95 = summarize([5, 1, 3, 2, 4])
    assert med == 3
    assert p95 == 5


def test_summarize_single_value():
    assert summarize([7.0]) == (7.0, 7.0)
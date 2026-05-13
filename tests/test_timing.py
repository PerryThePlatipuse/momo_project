import time

from text_distillation.timing import TimingTracker


def test_measure_records_positive_duration():
    tracker = TimingTracker()
    with tracker.measure("work"):
        time.sleep(0.01)
    timings = tracker.as_dict()
    assert "work" in timings
    assert timings["work"] >= 0.0


def test_repeated_measure_accumulates():
    tracker = TimingTracker()
    with tracker.measure("step"):
        time.sleep(0.005)
    first = tracker.as_dict()["step"]
    with tracker.measure("step"):
        time.sleep(0.005)
    second = tracker.as_dict()["step"]
    assert second > first


def test_as_dict_is_a_copy():
    tracker = TimingTracker()
    tracker.add("a", 1.0)
    snapshot = tracker.as_dict()
    snapshot["a"] = 999.0
    assert tracker["a"] == 1.0


def test_merge_combines_two_trackers():
    selection = TimingTracker()
    selection.add("selection_sec", 1.0)
    selection.add("embedding_forward_sec", 0.4)

    training = TimingTracker()
    training.add("training_sec", 2.0)
    training.merge(selection)

    combined = training.as_dict()
    assert combined == {"training_sec": 2.0, "selection_sec": 1.0, "embedding_forward_sec": 0.4}


def test_nested_measures_independent():
    tracker = TimingTracker()
    with tracker.measure("outer"):
        with tracker.measure("inner"):
            time.sleep(0.001)
    timings = tracker.as_dict()
    assert "outer" in timings and "inner" in timings
    assert timings["outer"] >= timings["inner"]

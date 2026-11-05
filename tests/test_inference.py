from dejargonizer.inference import _process_peak_memory_mb


def test_process_peak_memory_is_non_negative():
    assert _process_peak_memory_mb() >= 0

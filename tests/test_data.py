from dejargonizer.data import _quota_targets, split_records, stream_pilot


def test_split_records_is_deterministic_and_complete():
    records = [{"id": str(i), "source": f"source {i}"} for i in range(10)]
    first = split_records(records, seed=42)
    second = split_records(records, seed=42)
    assert first == second
    assert {record["id"] for record in first} == {str(i) for i in range(10)}


def test_stream_pilot_uses_streaming_and_stops_at_sample(monkeypatch):
    calls = []

    class FakeStream:
        def __iter__(self):
            for index in range(10):
                yield {"id": str(index), "abstract": "word " * 50}

        def shuffle(self, **kwargs):
            return self

    def fake_load_dataset(*args, **kwargs):
        calls.append(kwargs)
        return FakeStream()

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    config = {
        "dataset": {
            "name": "test",
            "split": "train",
            "revision": "main",
            "sample_size": 2,
            "max_scan_records": 5,
            "max_source_tokens": 100,
            "min_words": 2,
            "max_words": 100,
            "seed": 1,
            "shuffle_buffer": 2,
        }
    }
    assert len(stream_pilot(config)) == 2
    assert calls[0]["streaming"] is True


def test_chunk_quotas_round_to_requested_sample_size():
    targets = _quota_targets(100, [11.11] * 8 + [11.12])
    assert sum(targets) == 100
    assert max(targets) - min(targets) <= 1

from vgl.ops import TransformPipeline


def test_transform_pipeline_applies_transforms_in_order():
    calls = []

    def add_one(value):
        calls.append("add_one")
        return value + 1

    def times_two(value):
        calls.append("times_two")
        return value * 2

    pipeline = TransformPipeline((add_one, times_two))

    assert pipeline(3) == 8
    assert calls == ["add_one", "times_two"]


def test_transform_pipeline_append_returns_new_pipeline():
    pipeline = TransformPipeline()
    extended = pipeline.append(lambda value: value + 1)

    assert pipeline(4) == 4
    assert extended(4) == 5

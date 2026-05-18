from graspcorrect.vlm.openai_client import parse_points_json


def test_parse_points_json():
    assert parse_points_json('analysis {"points": [3, 7, 2]}')[:3] == [3, 7, 2]
    assert parse_points_json("points: [10, 1]")[:2] == [10, 1]

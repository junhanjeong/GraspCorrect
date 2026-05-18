from graspcorrect.vlm.base import parse_points_json


def test_parse_points_json_from_fenced_block():
    text = 'analysis\n```json\n{"points": [3, "7", 9]}\n```'
    assert parse_points_json(text) == [3, 7, 9]


def test_parse_points_json_from_loose_text():
    assert parse_points_json("I choose points 2, 5 and 8.") == [2, 5, 8]

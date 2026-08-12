import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from foundations.contracts import SceneFrame
from gaze_interaction.detector import CategoryFilter, YOLOEDetector


DEFAULT_FILTER = {
    "enabled": True,
    "categories": [
        {"name": "chair", "terms": ["chair", "armchair"]},
        {"name": "laptop", "terms": ["laptop", "notebook computer"]},
        {
            "name": "cellphone",
            "terms": [
                "cell phone",
                "cellphone",
                "mobile phone",
                "smartphone",
                "smart phone",
            ],
        },
        {"name": "tablet", "terms": ["ipad", "tablet", "tablet computer"]},
        {"name": "wall poster", "terms": ["wall poster", "poster", "placard"]},
    ],
}


@pytest.mark.parametrize(
    ("raw_label", "expected"),
    [
        ("office chair", "chair"),
        ("notebook computer", "laptop"),
        ("CELL-PHONE", "cellphone"),
        ("Apple iPad", "tablet"),
        ("decorative wall poster", "wall poster"),
        ("candle", None),
    ],
)
def test_category_filter_maps_phrases_without_substring_false_positives(
    raw_label: str, expected: str | None
) -> None:
    category_filter = CategoryFilter.from_config(DEFAULT_FILTER)

    assert category_filter.category_for(raw_label) == expected


def test_category_filter_prefers_the_most_specific_matching_term() -> None:
    category_filter = CategoryFilter.from_config(
        {
            "enabled": True,
            "categories": [
                {"name": "can", "terms": ["can"]},
                {"name": "soda can", "terms": ["soda can"]},
            ],
        }
    )

    assert category_filter.category_for("red soda can") == "soda can"
    assert category_filter.category_for("candle") is None


def test_disabled_category_filter_preserves_raw_labels() -> None:
    category_filter = CategoryFilter.from_config(
        {"enabled": False, "categories": DEFAULT_FILTER["categories"]}
    )

    assert category_filter.category_for("soda can") == "soda can"
    assert category_filter.category_for(None) is None


@pytest.mark.parametrize(
    ("config", "error_type", "message"),
    [
        (
            {"enabled": "yes", "categories": []},
            TypeError,
            "category_filter.enabled must be a bool",
        ),
        (
            {"enabled": True, "categories": []},
            ValueError,
            "must not be empty",
        ),
        (
            {
                "enabled": True,
                "categories": [{"name": "chair", "terms": ["chair"], "typo": 1}],
            },
            ValueError,
            "unknown category_filter.categories[0] key",
        ),
    ],
)
def test_category_filter_rejects_malformed_configuration(
    config: dict, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=re.escape(message)):
        CategoryFilter.from_config(config)


class _FakeTensor:
    def __init__(self, values) -> None:
        self._values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class _FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = _FakeTensor(
            [
                (0.0, 0.0, 10.0, 10.0),
                (10.0, 0.0, 20.0, 10.0),
                (20.0, 0.0, 30.0, 10.0),
            ]
        )
        self.conf = _FakeTensor((0.91, 0.88, 0.84))
        self.cls = _FakeTensor((0, 1, 2))

    def __len__(self) -> int:
        return 3


class _FakeModel:
    def __init__(self) -> None:
        self.predict_arguments = None

    def predict(self, **kwargs):
        self.predict_arguments = kwargs
        return [
            SimpleNamespace(
                boxes=_FakeBoxes(),
                names={0: "office chair", 1: "soda can", 2: "Apple iPad"},
            )
        ]


def test_detector_filters_and_canonicalizes_before_returning_detections(
    monkeypatch,
) -> None:
    model = _FakeModel()
    monkeypatch.setitem(
        sys.modules,
        "ultralytics",
        SimpleNamespace(YOLOE=lambda _source: model),
    )
    detector = YOLOEDetector(
        "fake.pt",
        confidence_threshold=0.45,
        image_size=640,
        device="cpu",
        category_filter=DEFAULT_FILTER,
    )
    frame = SceneFrame(0.0, np.zeros((20, 40, 3), dtype=np.uint8))

    detections = detector.detect(frame)

    assert [detection.label for detection in detections] == ["chair", "tablet"]
    assert [detection.confidence for detection in detections] == pytest.approx(
        [0.91, 0.84]
    )
    assert model.predict_arguments["conf"] == 0.45

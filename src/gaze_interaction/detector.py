"""Thin adapter around the Ultralytics prompt-free YOLOE detector."""

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

import numpy as np

from foundations.contracts import SceneFrame
from gaze_interaction.contracts import Detection, normalize_pixel_box


class CategoryFilter:
    """Map nuanced detector labels onto a configured category allowlist."""

    def __init__(
        self,
        *,
        enabled: bool,
        categories: Sequence[Mapping[str, object]],
    ) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("category_filter.enabled must be a bool")
        if isinstance(categories, (str, bytes)) or not isinstance(categories, Sequence):
            raise TypeError("category_filter.categories must be a list")

        rules: list[tuple[str, tuple[tuple[str, ...], ...]]] = []
        category_names: set[tuple[str, ...]] = set()
        terms_by_category: dict[tuple[str, ...], str] = {}
        for index, category in enumerate(categories):
            path = f"category_filter.categories[{index}]"
            if not isinstance(category, Mapping):
                raise TypeError(f"{path} must be a mapping")
            unknown_keys = set(category) - {"name", "terms"}
            missing_keys = {"name", "terms"} - set(category)
            if unknown_keys:
                unknown = ", ".join(sorted(str(key) for key in unknown_keys))
                raise ValueError(f"unknown {path} key(s): {unknown}")
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise ValueError(f"missing {path} key(s): {missing}")

            name = category["name"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{path}.name must be a non-empty string")
            normalized_name = _tokens(name)
            if not normalized_name:
                raise ValueError(f"{path}.name must contain letters or numbers")
            if normalized_name in category_names:
                raise ValueError(f"duplicate category name: {name}")
            category_names.add(normalized_name)

            raw_terms = category["terms"]
            if isinstance(raw_terms, (str, bytes)) or not isinstance(
                raw_terms, Sequence
            ):
                raise TypeError(f"{path}.terms must be a list")
            if not raw_terms:
                raise ValueError(f"{path}.terms must not be empty")

            normalized_terms: list[tuple[str, ...]] = []
            for term_index, term in enumerate(raw_terms):
                term_path = f"{path}.terms[{term_index}]"
                if not isinstance(term, str) or not term.strip():
                    raise ValueError(f"{term_path} must be a non-empty string")
                normalized_term = _tokens(term)
                if not normalized_term:
                    raise ValueError(f"{term_path} must contain letters or numbers")
                previous_category = terms_by_category.get(normalized_term)
                if previous_category is not None:
                    rendered = " ".join(normalized_term)
                    raise ValueError(
                        f"category term {rendered!r} is duplicated in "
                        f"{previous_category!r} and {name!r}"
                    )
                terms_by_category[normalized_term] = name
                normalized_terms.append(normalized_term)
            rules.append((name, tuple(normalized_terms)))

        if enabled and not rules:
            raise ValueError(
                "category_filter.categories must not be empty when filtering is enabled"
            )
        self.enabled = enabled
        self._rules = tuple(rules)

    @classmethod
    def from_config(cls, config: Mapping[str, object] | None) -> "CategoryFilter":
        if config is None:
            return cls(enabled=False, categories=())
        if not isinstance(config, Mapping):
            raise TypeError("category_filter must be a mapping")
        unknown_keys = set(config) - {"enabled", "categories"}
        missing_keys = {"enabled", "categories"} - set(config)
        if unknown_keys:
            unknown = ", ".join(sorted(str(key) for key in unknown_keys))
            raise ValueError(f"unknown category_filter key(s): {unknown}")
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"missing category_filter key(s): {missing}")
        return cls(
            enabled=config["enabled"],
            categories=config["categories"],
        )

    def category_for(self, label: str | None) -> str | None:
        """Return the canonical category for a raw label, or None if rejected."""

        if not self.enabled:
            return label
        if label is None:
            return None
        label_tokens = _tokens(label)
        best_match: tuple[int, int, str] | None = None
        for category_index, (category_name, terms) in enumerate(self._rules):
            for term in terms:
                if not _contains_phrase(label_tokens, term):
                    continue
                candidate = (len(term), -category_index, category_name)
                if best_match is None or candidate[:2] > best_match[:2]:
                    best_match = candidate
        return best_match[2] if best_match is not None else None


class YOLOEDetector:
    """Load YOLOE explicitly and expose only normalized local Detection records."""

    def __init__(
        self,
        model: str | Path,
        *,
        confidence_threshold: float,
        image_size: int,
        device: str,
        category_filter: Mapping[str, object] | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty string")

        # Keep imports and possible weight resolution/download out of module import/smoke paths.
        from ultralytics import YOLOE

        self.model_source = str(model)
        self.confidence_threshold = float(confidence_threshold)
        self.image_size = int(image_size)
        self.device = device
        self.category_filter = CategoryFilter.from_config(category_filter)
        self._model = YOLOE(self.model_source)

    def detect(self, frame: SceneFrame) -> tuple[Detection, ...]:
        if not isinstance(frame, SceneFrame):
            raise TypeError("frame must be a foundations.contracts.SceneFrame")

        # Foundation frames are RGB; Ultralytics treats ndarray sources as BGR.
        bgr_image = np.ascontiguousarray(frame.image[:, :, ::-1])
        results = self._model.predict(
            source=bgr_image,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results:
            return ()
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return ()

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        height, width = frame.image.shape[:2]
        detections: list[Detection] = []
        for coordinates, confidence, class_id in zip(
            xyxy, confidences, class_ids, strict=True
        ):
            label = self.category_filter.category_for(
                _class_name(result.names, int(class_id))
            )
            if self.category_filter.enabled and label is None:
                continue
            normalized_box = normalize_pixel_box(
                coordinates,
                image_width=width,
                image_height=height,
            )
            if normalized_box is None:
                continue
            detections.append(
                Detection(
                    box=normalized_box,
                    label=label,
                    confidence=float(confidence),
                )
            )
        return tuple(detections)


def _class_name(names: Any, class_id: int) -> str | None:
    if isinstance(names, dict):
        value = names.get(class_id)
    elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        value = names[class_id]
    else:
        value = None
    return str(value) if value is not None else None


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))


def _contains_phrase(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    width = len(phrase)
    return any(
        tokens[start : start + width] == phrase
        for start in range(len(tokens) - width + 1)
    )


if __name__ == "__main__":
    from gaze_interaction.contracts import normalize_pixel_box

    print(normalize_pixel_box((-1, 2, 11, 8), image_width=10, image_height=10))

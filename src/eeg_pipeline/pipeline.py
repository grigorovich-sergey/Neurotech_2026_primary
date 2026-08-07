"""Live/replay-compatible EEG buffer -> quality -> processing -> feature path."""

from eeg_pipeline.buffer import EEGBuffer
from eeg_pipeline.contracts import EEGFeatureWindow, EEGSample, EEGWindow, QualityState
from eeg_pipeline.processing import EEGFeatureExtractor, EEGPreprocessor, EEGQualityGate


class EEGPipeline:
    """Own ordered raw EEG and answer timestamp-addressed window/feature requests."""

    def __init__(
        self,
        *,
        buffer: EEGBuffer,
        preprocessor: EEGPreprocessor,
        quality_gate: EEGQualityGate,
        feature_extractor: EEGFeatureExtractor,
    ) -> None:
        self.buffer = buffer
        self.preprocessor = preprocessor
        self.quality_gate = quality_gate
        self.feature_extractor = feature_extractor

    def add_sample(self, sample: EEGSample) -> None:
        self.buffer.add(sample)

    def window(self, start: float, end: float) -> EEGWindow:
        return self.buffer.window(start, end)

    def features(self, start: float, end: float) -> EEGFeatureWindow:
        window = self.window(start, end)
        quality_state, quality_reasons = self.quality_gate.evaluate(window)
        values = None
        if quality_state is QualityState.USABLE:
            try:
                processed = self.preprocessor.process(window)
            except ValueError:
                quality_state = QualityState.UNAVAILABLE
                quality_reasons = (*quality_reasons, "filter_window_too_short")
            else:
                values = self.feature_extractor.extract(processed)
        return EEGFeatureWindow(
            requested_start=window.requested_start,
            requested_end=window.requested_end,
            actual_start=window.actual_start,
            actual_end=window.actual_end,
            sample_count=len(window.samples),
            completeness=window.completeness,
            quality_state=quality_state,
            quality_reasons=quality_reasons,
            feature_names=self.feature_extractor.feature_names,
            values=values,
        )


if __name__ == "__main__":
    from eeg_pipeline.synthetic import synthetic_eeg_samples

    sample_rate = 250.0
    pipeline = EEGPipeline(
        buffer=EEGBuffer(5.0),
        preprocessor=EEGPreprocessor(sample_rate_hz=sample_rate),
        quality_gate=EEGQualityGate(sample_rate_hz=sample_rate),
        feature_extractor=EEGFeatureExtractor(sample_rate_hz=sample_rate),
    )
    for sample in synthetic_eeg_samples(sample_rate_hz=sample_rate, duration_seconds=2.0):
        pipeline.add_sample(sample)
    print(pipeline.features(0.0, 2.0))

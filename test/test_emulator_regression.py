from pathlib import Path
import random

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import v2ecore.emulator as emulator_module
from v2ecore.emulator import EventEmulator


class _OutputSpy:
    """Record emulator adapter dispatch and closure counts."""

    def __init__(self, *_args_: object, **_kwargs_: object) -> None:
        self.append_count = 0
        self.close_count = 0

    def appendEvents(self, events_: np.ndarray, **_kwargs_: object) -> None:
        self.append_count += 1

    def close(self) -> None:
        self.close_count += 1


class _VideoWriterSpy:
    """Record video-writer release calls."""

    def __init__(self) -> None:
        self.release_count = 0

    def release(self) -> None:
        self.release_count += 1


def _Convert_events_to_h5_rows(events_: np.ndarray) -> np.ndarray:
    """Convert public event rows to the legacy HDF5 integer schema."""
    rows_ = np.array(events_, dtype=np.float32)
    rows_[:, 0] *= 1.0e6
    rows_[rows_[:, 3] == -1, 3] = 0
    return rows_.astype(np.uint32)


def _assert_event_packet_valid(events: np.ndarray, height: int, width: int) -> None:
    assert events is not None
    assert events.ndim == 2
    assert events.shape[1] == 4
    assert np.all(np.diff(events[:, 0]) >= 0)
    assert np.all(events[:, 1] >= 0)
    assert np.all(events[:, 1] < width)
    assert np.all(events[:, 2] >= 0)
    assert np.all(events[:, 2] < height)
    assert set(np.unique(events[:, 3])).issubset({-1.0, 1.0})


def test_generate_events_shape_bounds_and_monotonic_timestamps():
    height, width = 24, 32
    emu = EventEmulator(
        pos_thres=0.2,
        neg_thres=0.2,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        photoreceptor_noise=False,
        refractory_period_s=0.0,
        seed=7,
        output_width=width,
        output_height=height,
        device="cpu",
    )

    frame_0 = np.zeros((height, width), dtype=np.uint8)
    frame_1 = np.full((height, width), 255, dtype=np.uint8)

    assert emu.generate_events(frame_0, 0.0) is None
    events = emu.generate_events(frame_1, 1.0 / 30.0)
    emu.cleanup()

    _assert_event_packet_valid(events, height=height, width=width)


def test_strict_model_validity_rejects_undersampled_low_pass() -> None:
    """Direct emulator use fails before an invalid low-pass update is applied."""
    emulator_ = EventEmulator(
        pos_thres=0.2,
        neg_thres=0.2,
        sigma_thres=0.0,
        cutoff_hz=1.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        refractory_period_s=0.0,
        seed=71,
        output_width=4,
        output_height=4,
        device="cpu",
        strict_model_validity=True,
    )
    dark_frame_ = np.zeros((4, 4), dtype=np.uint8)
    bright_frame_ = np.full((4, 4), 255, dtype=np.uint8)

    try:
        assert emulator_.generate_events(dark_frame_, 0.0) is None
        with pytest.raises(ValueError, match=r"eps=.* > 1"):
            emulator_.generate_events(bright_frame_, 1.0)
    finally:
        emulator_.cleanup()


def _run_seeded_sequence(seed: int) -> np.ndarray:
    height, width = 16, 16
    emu = EventEmulator(
        pos_thres=0.12,
        neg_thres=0.12,
        sigma_thres=0.02,
        cutoff_hz=5.0,
        leak_rate_hz=0.01,
        shot_noise_rate_hz=0.02,
        photoreceptor_noise=False,
        refractory_period_s=0.0,
        seed=seed,
        output_width=width,
        output_height=height,
        device="cpu",
    )

    frames = []
    base = np.linspace(0, 255, width, dtype=np.float32)[None, :].repeat(
        height, axis=0)
    for i in range(6):
        frame = np.roll(base, i % 3, axis=1)
        frames.append(frame.astype(np.uint8))

    collected = []
    t = 0.0
    for frame in frames:
        events = emu.generate_events(frame, t)
        if events is not None:
            collected.append(events.copy())
        t += 1.0 / 60.0
    emu.cleanup()

    if len(collected) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(collected, axis=0)


def test_fixed_seed_reproducible_event_stream():
    stream_a = _run_seeded_sequence(seed=42)
    stream_b = _run_seeded_sequence(seed=42)
    assert np.array_equal(stream_a, stream_b)


def _run_seeded_photoreceptor_noise_sequence(seed: int) -> np.ndarray:
    height, width = 16, 16
    emu = EventEmulator(
        pos_thres=0.2,
        neg_thres=0.2,
        sigma_thres=0.03,
        cutoff_hz=30.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=200.0,
        photoreceptor_noise=True,
        refractory_period_s=0.0,
        seed=seed,
        output_width=width,
        output_height=height,
        device="cpu",
    )

    frame = np.full((height, width), 127, dtype=np.uint8)
    collected: list[np.ndarray] = []
    t = 0.0
    for _ in range(15):
        events = emu.generate_events(frame, t)
        if events is not None and events.shape[0] > 0:
            collected.append(events.copy())
        t += 1.0 / 120.0
    emu.cleanup()

    if len(collected) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(collected, axis=0)


def test_photoreceptor_noise_seed_reproducible_event_stream():
    stream_a = _run_seeded_photoreceptor_noise_sequence(seed=31)
    stream_b = _run_seeded_photoreceptor_noise_sequence(seed=31)
    assert np.array_equal(stream_a, stream_b)


def test_photoreceptor_noise_different_seeds_produce_different_streams():
    stream_a = _run_seeded_photoreceptor_noise_sequence(seed=41)
    stream_b = _run_seeded_photoreceptor_noise_sequence(seed=42)
    assert stream_a.shape[0] > 0
    assert stream_b.shape[0] > 0
    assert not np.array_equal(stream_a, stream_b)


def test_label_signal_noise_text_rows_match_returned_events(tmp_path: Path) -> None:
    """Labeled text output serializes every returned event exactly once."""
    height_ = 12
    width_ = 16
    text_name_ = "events-labeled.txt"

    emulator_ = EventEmulator(
        pos_thres=0.2,
        neg_thres=0.2,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=80.0,
        photoreceptor_noise=False,
        refractory_period_s=0.0,
        seed=11,
        output_folder=str(tmp_path),
        output_width=width_,
        output_height=height_,
        dvs_text=text_name_,
        label_signal_noise=True,
        device="cpu",
    )

    frame_ = np.full((height_, width_), 127, dtype=np.uint8)
    emulator_.generate_events(frame_, 0.0)
    events_ = emulator_.generate_events(frame_, 0.05)
    emulator_.cleanup()

    assert events_ is not None
    output_path_ = tmp_path / text_name_
    assert output_path_.exists()

    data_lines_ = [
        line_.strip() for line_ in output_path_.read_text().splitlines()
        if line_.strip() and not line_.startswith("#")
    ]
    assert len(data_lines_) == events_.shape[0]
    assert all(len(line_.split()) == 5 for line_ in data_lines_)


def test_moving_edge_generates_events_with_valid_packet():
    height, width = 32, 48
    emu = EventEmulator(
        pos_thres=0.15,
        neg_thres=0.15,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        photoreceptor_noise=False,
        refractory_period_s=0.0,
        seed=3,
        output_width=width,
        output_height=height,
        device="cpu",
    )

    # A rolled step edge creates both brightening and darkening transitions.
    x = np.arange(width, dtype=np.int32)[None, :].repeat(height, axis=0)
    frame_0 = np.where(x < (width // 2), 255, 0).astype(np.uint8)
    frame_1 = np.roll(frame_0, shift=4, axis=1)

    assert emu.generate_events(frame_0, 0.0) is None
    events = emu.generate_events(frame_1, 1.0 / 30.0)
    emu.cleanup()

    _assert_event_packet_valid(events, height=height, width=width)
    assert events.shape[0] > 0
    assert set(np.unique(events[:, 3])) == {-1.0, 1.0}


def test_moving_blob_generates_on_and_off_events():
    height, width = 40, 56
    emu = EventEmulator(
        pos_thres=0.12,
        neg_thres=0.12,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        photoreceptor_noise=False,
        refractory_period_s=0.0,
        seed=5,
        output_width=width,
        output_height=height,
        device="cpu",
    )

    frame_0 = np.zeros((height, width), dtype=np.uint8)
    frame_1 = np.zeros((height, width), dtype=np.uint8)
    blob_h, blob_w = 10, 12
    y0 = 12
    x0 = 8
    shift = 6
    frame_0[y0:y0 + blob_h, x0:x0 + blob_w] = 220
    frame_1[y0:y0 + blob_h, x0 + shift:x0 + shift + blob_w] = 220

    assert emu.generate_events(frame_0, 0.0) is None
    events = emu.generate_events(frame_1, 1.0 / 40.0)
    emu.cleanup()

    _assert_event_packet_valid(events, height=height, width=width)
    assert events.shape[0] > 0

    # Blob translation should create ON events on the leading edge and OFF on trailing edge.
    pol = set(np.unique(events[:, 3]))
    assert pol == {-1.0, 1.0}


def test_event_output_adapters_append_each_packet_once(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """Each enabled adapter receives one call for one returned event packet."""
    monkeypatch.setattr(emulator_module, "AEDat2Output", _OutputSpy)
    monkeypatch.setattr(emulator_module, "AEDat4Output", _OutputSpy)
    monkeypatch.setattr(emulator_module, "DVSTextOutput", _OutputSpy)

    emulator_ = EventEmulator(
        pos_thres=0.08,
        neg_thres=0.08,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        photoreceptor_noise=False,
        refractory_period_s=0.0,
        seed=18,
        output_folder=str(tmp_path),
        output_width=8,
        output_height=8,
        dvs_aedat2="events.aedat",
        dvs_aedat4="events.aedat4",
        dvs_text="events.txt",
        label_signal_noise=True,
        device="cpu",
    )

    dark_frame_ = np.zeros((8, 8), dtype=np.uint8)
    bright_frame_ = np.full((8, 8), 255, dtype=np.uint8)
    emulator_.generate_events(dark_frame_, 0.0)
    events_ = emulator_.generate_events(bright_frame_, 1.0 / 30.0)

    assert events_ is not None
    assert emulator_.dvs_aedat2.append_count == 1
    assert emulator_.dvs_aedat4.append_count == 1
    assert emulator_.dvs_text.append_count == 1
    emulator_.cleanup()


def test_cleanup_releases_each_output_resource_once(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated cleanup does not close or release output resources twice."""
    monkeypatch.setattr(emulator_module, "AEDat2Output", _OutputSpy)
    monkeypatch.setattr(emulator_module, "AEDat4Output", _OutputSpy)
    monkeypatch.setattr(emulator_module, "DVSTextOutput", _OutputSpy)

    emulator_ = EventEmulator(
        output_folder=str(tmp_path),
        output_width=8,
        output_height=8,
        dvs_aedat2="events.aedat",
        dvs_aedat4="events.aedat4",
        dvs_text="events.txt",
        device="cpu",
    )
    aedat2_spy_ = emulator_.dvs_aedat2
    aedat4_spy_ = emulator_.dvs_aedat4
    text_spy_ = emulator_.dvs_text
    video_spy_ = _VideoWriterSpy()
    emulator_.video_writers["state"] = video_spy_

    emulator_.cleanup()
    emulator_.cleanup()

    assert aedat2_spy_.close_count == 1
    assert aedat4_spy_.close_count == 1
    assert text_spy_.close_count == 1
    assert video_spy_.release_count == 1
    assert emulator_.dvs_aedat2 is None
    assert emulator_.dvs_aedat4 is None
    assert emulator_.dvs_text is None
    assert emulator_.video_writers == {}


def test_h5_writer_buffers_and_preserves_rows_on_cleanup(tmp_path: Path) -> None:
    """Final cleanup persists the buffered public event stream exactly once."""
    height_ = 8
    width_ = 8
    emulator_ = EventEmulator(
        pos_thres=0.08,
        neg_thres=0.08,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        photoreceptor_noise=False,
        refractory_period_s=0.0,
        seed=13,
        output_folder=str(tmp_path),
        output_width=width_,
        output_height=height_,
        dvs_h5="events.h5",
        device="cpu",
    )

    dark_frame_ = np.zeros((height_, width_), dtype=np.uint8)
    bright_frame_ = np.full((height_, width_), 255, dtype=np.uint8)
    assert emulator_.generate_events(dark_frame_, 0.0) is None
    events_ = emulator_.generate_events(bright_frame_, 1.0 / 30.0)

    assert events_ is not None
    before_cleanup_ = emulator_.h5_writer_stats()
    assert before_cleanup_["logical_event_count"] == events_.shape[0]
    assert before_cleanup_["buffered_event_count"] == events_.shape[0]
    assert before_cleanup_["written_event_count"] == 0
    assert emulator_.dvs_h5_dataset.shape[0] == 0

    emulator_.cleanup()
    after_cleanup_ = emulator_.h5_writer_stats()
    emulator_.cleanup()

    assert after_cleanup_["buffered_event_count"] == 0
    assert after_cleanup_["written_event_count"] == events_.shape[0]
    assert after_cleanup_["flush_count"] == 1
    assert emulator_.h5_writer_stats() == after_cleanup_
    with h5py.File(tmp_path / "events.h5", "r") as h5_file_:
        np.testing.assert_array_equal(
            h5_file_["events"][:],
            _Convert_events_to_h5_rows(events_),
        )


def test_h5_writer_flushes_when_buffer_limit_is_reached(tmp_path: Path) -> None:
    """A full buffer is persisted before final cleanup without changing rows."""
    emulator_ = EventEmulator(
        pos_thres=0.08,
        neg_thres=0.08,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        photoreceptor_noise=False,
        refractory_period_s=0.0,
        seed=14,
        output_folder=str(tmp_path),
        output_width=4,
        output_height=4,
        dvs_h5="events.h5",
        device="cpu",
    )
    emulator_.H5_EVENT_BUFFER_ROWS = 1

    dark_frame_ = np.zeros((4, 4), dtype=np.uint8)
    bright_frame_ = np.full((4, 4), 255, dtype=np.uint8)
    emulator_.generate_events(dark_frame_, 0.0)
    events_ = emulator_.generate_events(bright_frame_, 1.0 / 30.0)

    assert events_ is not None
    stats_ = emulator_.h5_writer_stats()
    assert stats_["written_event_count"] == events_.shape[0]
    assert stats_["buffered_event_count"] == 0
    assert stats_["flush_count"] == 1
    np.testing.assert_array_equal(
        emulator_.dvs_h5_dataset[:],
        _Convert_events_to_h5_rows(events_),
    )
    emulator_.cleanup()


def test_h5_frame_idx_counts_buffered_events(tmp_path: Path) -> None:
    """Frame metadata includes events not yet physically written to HDF5."""
    emulator_ = EventEmulator(
        pos_thres=0.08,
        neg_thres=0.08,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        photoreceptor_noise=False,
        refractory_period_s=0.0,
        seed=15,
        output_folder=str(tmp_path),
        output_width=4,
        output_height=4,
        dvs_h5="events.h5",
        device="cpu",
    )
    emulator_.prepare_storage(2, [0.0, 1.0 / 30.0])

    dark_frame_ = np.zeros((4, 4), dtype=np.uint8)
    bright_frame_ = np.full((4, 4), 255, dtype=np.uint8)
    emulator_.generate_events(dark_frame_, 0.0)
    events_ = emulator_.generate_events(bright_frame_, 1.0 / 30.0)

    assert events_ is not None
    assert emulator_.dvs_h5_dataset.shape[0] == 0
    assert int(emulator_.frame_ev_idx_dataset[1]) == events_.shape[0]
    emulator_.cleanup()


def test_iebcs_latency_jitter_model_delays_timestamps_and_keeps_monotonic():
    height, width = 24, 24
    common_kwargs = dict(
        pos_thres=0.2,
        neg_thres=0.2,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        photoreceptor_noise=False,
        refractory_period_s=0.0,
        seed=9,
        output_width=width,
        output_height=height,
        device="cpu",
    )

    frame_0 = np.zeros((height, width), dtype=np.uint8)
    frame_1 = np.full((height, width), 255, dtype=np.uint8)

    emu_base = EventEmulator(
        **common_kwargs,
        iebcs_latency_jitter_model=False,
    )
    emu_base.generate_events(frame_0, 0.0)
    events_base = emu_base.generate_events(frame_1, 1.0 / 30.0)
    emu_base.cleanup()

    emu_latency = EventEmulator(
        **common_kwargs,
        iebcs_latency_jitter_model=True,
        iebcs_latency_mean_us=200.0,
        iebcs_latency_jitter_us=0.0,
    )
    emu_latency.generate_events(frame_0, 0.0)
    events_latency = emu_latency.generate_events(frame_1, 1.0 / 30.0)
    emu_latency.cleanup()

    assert events_base is not None and events_latency is not None
    assert events_base.shape == events_latency.shape
    assert np.all(np.diff(events_latency[:, 0]) >= 0)

    base_ts = np.sort(events_base[:, 0])
    latency_ts = np.sort(events_latency[:, 0])
    assert np.allclose(latency_ts - base_ts, 200e-6, atol=1e-8)


def test_iebcs_resample_thresholds_on_event_updates_thresholds_only_when_enabled():
    height, width = 20, 20
    common_kwargs = dict(
        pos_thres=0.2,
        neg_thres=0.2,
        sigma_thres=0.03,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        photoreceptor_noise=False,
        refractory_period_s=0.0,
        seed=17,
        output_width=width,
        output_height=height,
        device="cpu",
    )

    frame_0 = np.zeros((height, width), dtype=np.uint8)
    frame_1 = np.full((height, width), 255, dtype=np.uint8)

    emu_disabled = EventEmulator(
        **common_kwargs,
        iebcs_resample_thresholds_on_event=False,
    )
    emu_disabled.generate_events(frame_0, 0.0)
    pos_before_disabled = emu_disabled.pos_thres.clone()
    neg_before_disabled = emu_disabled.neg_thres.clone()
    events_disabled = emu_disabled.generate_events(frame_1, 1.0 / 30.0)
    pos_after_disabled = emu_disabled.pos_thres.clone()
    neg_after_disabled = emu_disabled.neg_thres.clone()
    emu_disabled.cleanup()

    emu_enabled = EventEmulator(
        **common_kwargs,
        iebcs_resample_thresholds_on_event=True,
    )
    emu_enabled.generate_events(frame_0, 0.0)
    pos_before_enabled = emu_enabled.pos_thres.clone()
    neg_before_enabled = emu_enabled.neg_thres.clone()
    events_enabled = emu_enabled.generate_events(frame_1, 1.0 / 30.0)
    pos_after_enabled = emu_enabled.pos_thres.clone()
    neg_after_enabled = emu_enabled.neg_thres.clone()
    emu_enabled.cleanup()

    assert events_disabled is not None and events_disabled.shape[0] > 0
    assert events_enabled is not None and events_enabled.shape[0] > 0

    assert torch.equal(pos_before_disabled, pos_after_disabled)
    assert torch.equal(neg_before_disabled, neg_after_disabled)

    pos_changed = torch.any(~torch.isclose(pos_before_enabled, pos_after_enabled))
    neg_changed = torch.any(~torch.isclose(neg_before_enabled, neg_after_enabled))
    assert bool(pos_changed or neg_changed)


def _run_burst_timestamp_mode(*,
                              nonuniform_enabled: bool,
                              mode: str,
                              seed: int) -> np.ndarray:
    height, width = 18, 18
    emu = EventEmulator(
        pos_thres=0.08,
        neg_thres=0.08,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        photoreceptor_noise=False,
        refractory_period_s=0.0,
        v2ce_nonuniform_burst_timestamps=nonuniform_enabled,
        v2ce_burst_timestamps_mode=mode,
        seed=seed,
        output_width=width,
        output_height=height,
        device="cpu",
    )
    frame_0 = np.zeros((height, width), dtype=np.uint8)
    frame_1 = np.full((height, width), 255, dtype=np.uint8)
    emu.generate_events(frame_0, 0.0)
    events = emu.generate_events(frame_1, 1.0 / 30.0)
    emu.cleanup()
    assert events is not None
    return events


def test_v2ce_random_burst_timestamps_are_nonuniform_and_monotonic():
    events_linear = _run_burst_timestamp_mode(
        nonuniform_enabled=False,
        mode="random",
        seed=23)
    events_random = _run_burst_timestamp_mode(
        nonuniform_enabled=True,
        mode="random",
        seed=23)

    assert events_linear.shape[0] == events_random.shape[0]
    assert np.all(np.diff(events_random[:, 0]) >= 0)

    linear_unique_ts = np.unique(events_linear[:, 0])
    random_unique_ts = np.unique(events_random[:, 0])
    assert len(linear_unique_ts) > 1
    assert len(random_unique_ts) == len(linear_unique_ts)
    assert not np.allclose(random_unique_ts, linear_unique_ts, atol=1e-8)


def test_v2ce_slope_mode_biases_events_later_than_random_mode():
    events_random = _run_burst_timestamp_mode(
        nonuniform_enabled=True,
        mode="random",
        seed=31)
    events_slope = _run_burst_timestamp_mode(
        nonuniform_enabled=True,
        mode="slope",
        seed=31)

    assert events_random.shape[0] == events_slope.shape[0]
    assert np.all(np.diff(events_slope[:, 0]) >= 0)
    assert float(np.mean(events_slope[:, 0])) > float(np.mean(events_random[:, 0]))


def _Write_hist_noise_file(path, bins: int = 8) -> None:
    cdf = np.zeros((4, bins), dtype=np.float32)
    step_idx = min(4, bins - 1)
    cdf[:, step_idx:] = 1.0
    np.save(path, cdf)


def test_contrast_latency_model_increases_delay_for_low_slope_cases():
    height, width = 24, 24
    common = dict(
        pos_thres=0.05,
        neg_thres=0.05,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        photoreceptor_noise=False,
        refractory_period_s=0.0,
        iebcs_contrast_latency_model=True,
        iebcs_latency_mean_us=100.0,
        iebcs_latency_jitter_us=0.0,
        iebcs_latency_tau_us=300.0,
        output_width=width,
        output_height=height,
        device="cpu",
    )

    frame_0 = np.full((height, width), 40, dtype=np.uint8)
    frame_low = np.full((height, width), 90, dtype=np.uint8)
    frame_high = np.full((height, width), 255, dtype=np.uint8)

    emu_low = EventEmulator(seed=101, **common)
    emu_low.generate_events(frame_0, 0.0)
    events_low = emu_low.generate_events(frame_low, 1.0 / 30.0)
    emu_low.cleanup()

    emu_high = EventEmulator(seed=101, **common)
    emu_high.generate_events(frame_0, 0.0)
    events_high = emu_high.generate_events(frame_high, 1.0 / 30.0)
    emu_high.cleanup()

    assert events_low is not None and events_low.shape[0] > 0
    assert events_high is not None and events_high.shape[0] > 0
    assert float(np.mean(events_low[:, 0])) > float(np.mean(events_high[:, 0]))


def test_contrast_latency_model_keeps_monotonic_timestamps():
    height, width = 20, 20
    emu = EventEmulator(
        pos_thres=0.05,
        neg_thres=0.05,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        photoreceptor_noise=False,
        refractory_period_s=0.0,
        iebcs_contrast_latency_model=True,
        iebcs_latency_mean_us=200.0,
        iebcs_latency_jitter_us=20.0,
        iebcs_latency_tau_us=300.0,
        seed=41,
        output_width=width,
        output_height=height,
        device="cpu",
    )
    frame_0 = np.zeros((height, width), dtype=np.uint8)
    frame_1 = np.full((height, width), 255, dtype=np.uint8)
    emu.generate_events(frame_0, 0.0)
    events = emu.generate_events(frame_1, 1.0 / 30.0)
    emu.cleanup()
    assert events is not None and events.shape[0] > 0
    assert np.all(np.diff(events[:, 0]) >= 0)


def test_hist_noise_model_emits_events_without_signal_changes(tmp_path):
    pos_path = tmp_path / "pos_hist.npy"
    neg_path = tmp_path / "neg_hist.npy"
    _Write_hist_noise_file(pos_path, bins=8)
    _Write_hist_noise_file(neg_path, bins=8)

    height, width = 12, 12
    emu = EventEmulator(
        pos_thres=0.2,
        neg_thres=0.2,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        photoreceptor_noise=False,
        refractory_period_s=0.0,
        iebcs_hist_noise_model=True,
        iebcs_hist_noise_pos_path=str(pos_path),
        iebcs_hist_noise_neg_path=str(neg_path),
        seed=5,
        output_width=width,
        output_height=height,
        device="cpu",
    )

    frame = np.full((height, width), 120, dtype=np.uint8)
    emu.generate_events(frame, 0.0)
    events = emu.generate_events(frame, 0.2)
    emu.cleanup()

    assert events is not None
    assert events.shape[0] > 0
    _assert_event_packet_valid(events, height=height, width=width)


def test_hist_noise_files_mode_requires_paths():
    with pytest.raises(ValueError):
        EventEmulator(
            iebcs_hist_noise_model=True,
            iebcs_hist_noise_pos_path=None,
            iebcs_hist_noise_neg_path=None,
            output_width=8,
            output_height=8,
            device="cpu",
        )


def test_refractory_state_coupling_reduces_illegal_burst_events():
    height, width = 18, 18
    frame_0 = np.zeros((height, width), dtype=np.uint8)
    frame_1 = np.full((height, width), 255, dtype=np.uint8)

    emu_base = EventEmulator(
        pos_thres=0.05,
        neg_thres=0.05,
        sigma_thres=0.0,
        refractory_period_s=0.0,
        seed=77,
        output_width=width,
        output_height=height,
        device="cpu",
    )
    emu_base.generate_events(frame_0, 0.0)
    events_base = emu_base.generate_events(frame_1, 1.0 / 30.0)
    emu_base.cleanup()

    emu_coupled = EventEmulator(
        pos_thres=0.05,
        neg_thres=0.05,
        sigma_thres=0.0,
        refractory_period_s=0.0,
        iebcs_refractory_state_coupling=True,
        iebcs_refractory_us=12000.0,
        seed=77,
        output_width=width,
        output_height=height,
        device="cpu",
    )
    emu_coupled.generate_events(frame_0, 0.0)
    events_coupled = emu_coupled.generate_events(frame_1, 1.0 / 30.0)
    emu_coupled.cleanup()

    assert events_base is not None and events_coupled is not None
    assert events_base.shape[0] > events_coupled.shape[0]


def test_refractory_state_coupling_preserves_packet_validity_and_order():
    height, width = 16, 16
    emu = EventEmulator(
        pos_thres=0.05,
        neg_thres=0.05,
        sigma_thres=0.0,
        refractory_period_s=0.0,
        iebcs_refractory_state_coupling=True,
        iebcs_refractory_us=10000.0,
        seed=99,
        output_width=width,
        output_height=height,
        device="cpu",
    )
    frame_0 = np.zeros((height, width), dtype=np.uint8)
    frame_1 = np.full((height, width), 255, dtype=np.uint8)
    emu.generate_events(frame_0, 0.0)
    events = emu.generate_events(frame_1, 1.0 / 25.0)
    emu.cleanup()

    _assert_event_packet_valid(events, height=height, width=width)


def test_refractory_state_coupling_with_contrast_latency_model_runs_without_dtype_errors():
    height, width = 16, 16
    emu = EventEmulator(
        pos_thres=0.05,
        neg_thres=0.05,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        photoreceptor_noise=False,
        refractory_period_s=0.0,
        iebcs_contrast_latency_model=True,
        iebcs_latency_mean_us=200.0,
        iebcs_latency_jitter_us=20.0,
        iebcs_latency_tau_us=300.0,
        iebcs_refractory_state_coupling=True,
        iebcs_refractory_us=500.0,
        seed=101,
        output_width=width,
        output_height=height,
        device="cpu",
    )
    frame_0 = np.zeros((height, width), dtype=np.uint8)
    frame_1 = np.full((height, width), 255, dtype=np.uint8)
    emu.generate_events(frame_0, 0.0)
    events = emu.generate_events(frame_1, 1.0 / 30.0)
    emu.cleanup()

    assert events is not None and events.shape[0] > 0
    _assert_event_packet_valid(events, height=height, width=width)
    assert np.all(np.diff(events[:, 0]) >= 0)


def test_new_models_disabled_matches_baseline_behavior():
    height, width = 14, 14
    frame_0 = np.zeros((height, width), dtype=np.uint8)
    frame_1 = np.full((height, width), 180, dtype=np.uint8)
    common = dict(
        pos_thres=0.15,
        neg_thres=0.15,
        sigma_thres=0.02,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        photoreceptor_noise=False,
        seed=123,
        output_width=width,
        output_height=height,
        device="cpu",
    )
    emu_base = EventEmulator(**common)
    emu_base.generate_events(frame_0, 0.0)
    ev_base = emu_base.generate_events(frame_1, 1.0 / 30.0)
    emu_base.cleanup()

    emu_disabled = EventEmulator(
        **common,
        iebcs_contrast_latency_model=False,
        iebcs_hist_noise_model=False,
        iebcs_refractory_state_coupling=False,
    )
    emu_disabled.generate_events(frame_0, 0.0)
    ev_disabled = emu_disabled.generate_events(frame_1, 1.0 / 30.0)
    emu_disabled.cleanup()

    assert np.array_equal(ev_base, ev_disabled)


def test_refractory_release_interpolation_is_idempotent_within_frame():
    emu = EventEmulator(
        output_width=2,
        output_height=2,
        device="cpu",
        iebcs_refractory_state_coupling=True,
        refractory_period_s=1e-3,
    )
    emu.base_log_frame = torch.zeros((2, 2), dtype=torch.float32)
    emu.photoreceptor_noise_arr = torch.zeros((2, 2), dtype=torch.float32)
    emu.iebcs_refractory_release_ts = torch.tensor(
        [[0.005, 0.0], [0.0, 0.0]], dtype=torch.float32)
    emu.t_previous = 0.0
    photoreceptor = torch.ones((2, 2), dtype=torch.float32)
    ts = torch.tensor(0.01, dtype=torch.float32)

    emu._apply_refractory_release_interpolation(
        ts=ts, delta_time=0.01, photoreceptor=photoreceptor)
    first = emu.base_log_frame.clone()
    emu._apply_refractory_release_interpolation(
        ts=ts, delta_time=0.01, photoreceptor=photoreceptor)
    second = emu.base_log_frame.clone()
    emu.cleanup()

    assert torch.equal(first, second)


def test_hist_noise_event_generation_is_capped_and_reschedules_dropped_due_events(caplog: pytest.LogCaptureFixture):
    emu = EventEmulator(
        output_width=1,
        output_height=1,
        device="cpu",
    )
    emu.iebcs_hist_noise_model = True
    emu.iebcs_noise_bins_hz = torch.tensor([1e9, 1e9], dtype=torch.float32)
    emu.iebcs_noise_cdf_pos = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    emu.iebcs_noise_cdf_neg = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    emu.iebcs_noise_idx_pos = torch.zeros((1, 1), dtype=torch.int64)
    emu.iebcs_noise_idx_neg = torch.zeros((1, 1), dtype=torch.int64)
    emu.iebcs_noise_next_pos_s = torch.zeros((1, 1), dtype=torch.float32)
    emu.iebcs_noise_next_neg_s = torch.zeros((1, 1), dtype=torch.float32)

    with caplog.at_level("WARNING", logger="v2ecore.emulator"):
        events = emu._sample_hist_noise_events(t_frame=0.1)
    emu.cleanup()

    cap = EventEmulator.IEBCS_MAX_NOISE_EVENTS_PER_FRAME_FACTOR * 1
    assert events.shape[0] <= cap
    assert float(emu.iebcs_noise_next_pos_s[0, 0]) > 0.1
    assert float(emu.iebcs_noise_next_neg_s[0, 0]) > 0.1
    assert any("capped" in rec.getMessage() for rec in caplog.records)


def test_reset_replays_a_seeded_sequence_from_time_zero() -> None:
    height_, width_ = 8, 10
    frame_0_ = np.zeros((height_, width_), dtype=np.uint8)
    frame_1_ = np.full((height_, width_), 220, dtype=np.uint8)
    emulator_kwargs_ = {
        "pos_thres": 0.08,
        "neg_thres": 0.08,
        "sigma_thres": 0.02,
        "cutoff_hz": 0.0,
        "leak_rate_hz": 0.1,
        "shot_noise_rate_hz": 0.0,
        "refractory_period_s": 0.0,
        "v2ce_nonuniform_burst_timestamps": True,
        "seed": 73,
        "output_width": width_,
        "output_height": height_,
        "device": "cpu",
    }

    reused_emulator_ = EventEmulator(**emulator_kwargs_)
    fresh_emulator_ = EventEmulator(**emulator_kwargs_)
    try:
        reused_emulator_.generate_events(frame_0_, 0.0)
        first_events_ = reused_emulator_.generate_events(frame_1_, 0.02)
        first_thresholds_ = reused_emulator_.pos_thres.clone()

        reused_emulator_.reset()
        assert reused_emulator_.t_previous == 0.0
        assert reused_emulator_.pos_thres == reused_emulator_.pos_thres_nominal

        reused_emulator_.generate_events(frame_0_, 0.0)
        replayed_events_ = reused_emulator_.generate_events(frame_1_, 0.02)
        replayed_thresholds_ = reused_emulator_.pos_thres.clone()

        fresh_emulator_.generate_events(frame_0_, 0.0)
        fresh_events_ = fresh_emulator_.generate_events(frame_1_, 0.02)
        fresh_thresholds_ = fresh_emulator_.pos_thres.clone()
    finally:
        reused_emulator_.cleanup()
        fresh_emulator_.cleanup()

    assert first_events_ is not None
    assert replayed_events_ is not None
    assert fresh_events_ is not None
    assert np.array_equal(first_events_, replayed_events_)
    assert np.array_equal(replayed_events_, fresh_events_)
    assert torch.equal(first_thresholds_, replayed_thresholds_)
    assert torch.equal(replayed_thresholds_, fresh_thresholds_)


def test_set_dvs_params_rejects_an_initialized_emulator() -> None:
    emulator_ = EventEmulator(
        output_width=4,
        output_height=4,
        device="cpu",
    )
    try:
        emulator_.generate_events(np.zeros((4, 4), dtype=np.uint8), 0.0)

        with pytest.raises(RuntimeError, match="reset.*before changing"):
            emulator_.set_dvs_params("noisy")
    finally:
        emulator_.cleanup()


def test_set_dvs_params_rebuilds_nominal_and_pixel_threshold_state() -> None:
    frame_ = np.full((6, 6), 80, dtype=np.uint8)
    emulator_ = EventEmulator(
        pos_thres=0.35,
        neg_thres=0.4,
        sigma_thres=0.01,
        leak_rate_hz=0.0,
        output_width=6,
        output_height=6,
        device="cpu",
        seed=19,
    )
    try:
        emulator_.set_dvs_params("clean")
        emulator_.generate_events(frame_, 0.0)

        assert emulator_.pos_thres_nominal == pytest.approx(0.2)
        assert emulator_.neg_thres_nominal == pytest.approx(0.2)
        assert isinstance(emulator_.pos_thres, torch.Tensor)
        assert isinstance(emulator_.neg_thres, torch.Tensor)
        assert torch.allclose(
            emulator_.pos_thres_pre_prob,
            emulator_.pos_thres_nominal / emulator_.pos_thres,
        )
        assert torch.allclose(
            emulator_.neg_thres_pre_prob,
            emulator_.neg_thres_nominal / emulator_.neg_thres,
        )

        emulator_.reset()
        emulator_.set_dvs_params("noisy")
        emulator_.generate_events(frame_, 0.0)
        assert emulator_.noise_rate_array is not None
        assert isinstance(emulator_.pos_thres, torch.Tensor)
        assert torch.allclose(
            emulator_.pos_thres_pre_prob,
            emulator_.pos_thres_nominal / emulator_.pos_thres,
        )
    finally:
        emulator_.cleanup()


def _Numpy_rng_states_equal(first_: tuple[object, ...], second_: tuple[object, ...]) -> bool:
    """Return true when two legacy NumPy RNG state tuples are identical."""

    return (
        first_[0] == second_[0]
        and np.array_equal(first_[1], second_[1])
        and first_[2:] == second_[2:]
    )


def _Run_private_rng_pair(reverse_order_: bool,
                          include_second_: bool = True) -> dict[str, np.ndarray]:
    """Run one or two stochastic emulators with optionally reversed order."""

    height_, width_ = 10, 12
    common_ = {
        "pos_thres": 0.12,
        "neg_thres": 0.12,
        "sigma_thres": 0.02,
        "cutoff_hz": 8.0,
        "leak_rate_hz": 0.2,
        "shot_noise_rate_hz": 4.0,
        "refractory_period_s": 0.0,
        "iebcs_latency_jitter_model": True,
        "iebcs_resample_thresholds_on_event": True,
        "v2ce_nonuniform_burst_timestamps": True,
        "output_width": width_,
        "output_height": height_,
        "device": "cpu",
    }
    emulators_ = {
        "a": EventEmulator(seed=101, **common_),
    }
    if include_second_:
        emulators_["b"] = EventEmulator(seed=202, **common_)
    frames_ = tuple(
        np.roll(
            np.linspace(10, 245, width_, dtype=np.uint8)[None, :].repeat(height_, axis=0),
            frame_index_,
            axis=1,
        )
        for frame_index_ in range(5)
    )
    chunks_: dict[str, list[np.ndarray]] = {name_: [] for name_ in emulators_}
    names_ = tuple(reversed(tuple(emulators_))) if reverse_order_ else tuple(emulators_)
    try:
        for frame_index_, frame_ in enumerate(frames_):
            for name_ in names_:
                events_ = emulators_[name_].generate_events(frame_, frame_index_ / 50.0)
                if events_ is not None and events_.size:
                    chunks_[name_].append(events_.copy())
    finally:
        for emulator_ in emulators_.values():
            emulator_.cleanup()

    return {
        name_: np.concatenate(stream_chunks_, axis=0) if stream_chunks_ else np.empty((0, 4), dtype=np.float32)
        for name_, stream_chunks_ in chunks_.items()
    }


def test_seeded_emulator_does_not_mutate_ambient_random_states() -> None:
    random.seed(9001)
    np.random.seed(9002)
    torch.manual_seed(9003)
    python_state_ = random.getstate()
    numpy_state_ = np.random.get_state()
    torch_state_ = torch.random.get_rng_state().clone()
    cuda_states_ = tuple(state_.clone() for state_ in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()

    emulator_ = EventEmulator(
        seed=77,
        sigma_thres=0.03,
        leak_rate_hz=0.2,
        shot_noise_rate_hz=3.0,
        output_width=8,
        output_height=8,
        device="cpu",
    )
    try:
        emulator_.generate_events(np.zeros((8, 8), dtype=np.uint8), 0.0)
        emulator_.generate_events(np.full((8, 8), 180, dtype=np.uint8), 0.02)
    finally:
        emulator_.cleanup()

    assert random.getstate() == python_state_
    assert _Numpy_rng_states_equal(np.random.get_state(), numpy_state_)
    assert torch.equal(torch.random.get_rng_state(), torch_state_)
    if cuda_states_:
        assert all(
            torch.equal(current_, expected_)
            for current_, expected_ in zip(torch.cuda.get_rng_state_all(), cuda_states_, strict=True)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for device-generator isolation")
def test_seeded_cuda_emulator_does_not_mutate_ambient_cuda_random_state() -> None:
    cuda_states_ = tuple(state_.clone() for state_ in torch.cuda.get_rng_state_all())
    emulator_ = EventEmulator(
        seed=88,
        sigma_thres=0.03,
        leak_rate_hz=0.2,
        shot_noise_rate_hz=3.0,
        output_width=4,
        output_height=4,
        device="cuda:0",
    )
    try:
        emulator_.generate_events(np.zeros((4, 4), dtype=np.uint8), 0.0)
        emulator_.generate_events(np.full((4, 4), 180, dtype=np.uint8), 0.02)
        torch.cuda.synchronize(0)
    finally:
        emulator_.cleanup()

    assert all(
        torch.equal(current_, expected_)
        for current_, expected_ in zip(torch.cuda.get_rng_state_all(), cuda_states_, strict=True)
    )


def test_per_instance_generators_make_streams_independent_of_processing_order() -> None:
    forward_ = _Run_private_rng_pair(reverse_order_=False)
    reversed_ = _Run_private_rng_pair(reverse_order_=True)

    assert np.array_equal(forward_["a"], reversed_["a"])
    assert np.array_equal(forward_["b"], reversed_["b"])


def test_seeded_stream_matches_when_run_alone_or_with_another_emulator() -> None:
    alone_ = _Run_private_rng_pair(reverse_order_=False, include_second_=False)
    together_ = _Run_private_rng_pair(reverse_order_=False, include_second_=True)

    assert np.array_equal(alone_["a"], together_["a"])

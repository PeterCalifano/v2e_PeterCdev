import numpy as np

from v2ecore.emulator import EventEmulator


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


def test_label_signal_noise_shot_noise_path_writes_label_column(tmp_path):
    height, width = 12, 16
    text_name = "events-labeled.txt"

    emu = EventEmulator(
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
        output_width=width,
        output_height=height,
        dvs_text=text_name,
        label_signal_noise=True,
        device="cpu",
    )

    frame = np.full((height, width), 127, dtype=np.uint8)
    emu.generate_events(frame, 0.0)
    emu.generate_events(frame, 0.05)
    emu.cleanup()

    out_path = tmp_path / text_name
    assert out_path.exists()

    data_lines = [
        line.strip() for line in out_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(data_lines) > 0
    assert all(len(line.split()) == 5 for line in data_lines[:20])


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

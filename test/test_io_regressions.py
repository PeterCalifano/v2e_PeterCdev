import argparse
import importlib
import importlib.util
import logging
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

pytest.importorskip("easygui")

from v2ecore.v2e_utils import ImageFolderReader
from v2ecore.v2e_utils import read_aedat_txt_events
from v2ecore.v2e_utils import set_output_dimension
from v2ecore.v2e_utils import set_output_folder


def _Write_test_image(path: Path, pixel_value: int) -> None:
    image = np.full((6, 6, 3), pixel_value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _Build_v2e_args(argument_list: list[str]):
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    v2e_args_module = importlib.import_module("v2ecore.v2e_args")
    parser = argparse.ArgumentParser(formatter_class=v2e_args_module.SmartFormatter)
    parser = v2e_args_module.v2e_args(parser)
    return parser.parse_args(argument_list)


def _Import_local_v2e_module():
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    local_v2e_path = Path(__file__).resolve().parents[1] / "v2e.py"
    spec = importlib.util.spec_from_file_location(
        "v2e_local_under_test", str(local_v2e_path))
    assert spec is not None and spec.loader is not None
    v2e_local_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2e_local_module)
    return v2e_local_module


def test_set_output_folder_rejects_output_in_place_without_input_path():
    with pytest.raises(ValueError, match="output_in_place=True requires a real input file or input folder path"):
        set_output_folder(
            output_folder=None,
            input_file=None,
            unique_output_folder=True,
            overwrite=False,
            output_in_place=True,
            logger=logging.getLogger(__name__),
        )


def test_image_folder_reader_returns_false_at_end_of_sequence(tmp_path: Path):
    image_folder = tmp_path / "frames"
    image_folder.mkdir()
    _Write_test_image(image_folder / "00000000.png", pixel_value=42)

    reader = ImageFolderReader(str(image_folder), frame_rate=30.0)
    first_ret, first_frame = reader.read()
    second_ret, second_frame = reader.read()

    assert first_ret is True
    assert first_frame is not None
    assert second_ret is False
    assert second_frame is None


def test_image_folder_reader_preserves_float32_tiff_frames(tmp_path: Path) -> None:
    image_folder_ = tmp_path / "float_frames"
    image_folder_.mkdir()
    source_frame_ = np.linspace(
        0.0, 1.0 / 512.0, 36, dtype=np.float32).reshape(6, 6)
    assert cv2.imwrite(
        str(image_folder_ / "00000000.tiff"), source_frame_)

    reader_ = ImageFolderReader(str(image_folder_), frame_rate=30.0)
    ret_, frame_ = reader_.read()

    assert ret_ is True
    assert frame_ is not None
    assert frame_.dtype == np.float32
    assert frame_.ndim == 2
    np.testing.assert_allclose(frame_, source_frame_, atol=1.0e-7)


def test_main_no_slomo_hdr_matches_direct_float_input(monkeypatch: pytest.MonkeyPatch,
                                                       tmp_path: Path) -> None:
    """The no-SloMo CLI must preserve float frames and direct-model output."""
    h5py_ = pytest.importorskip("h5py")
    emulator_module_ = importlib.import_module("v2ecore.emulator")
    v2e_module_ = _Import_local_v2e_module()

    input_folder_ = tmp_path / "float_hdr_frames"
    input_folder_.mkdir()
    output_folder_ = tmp_path / "float_hdr_output"
    height_, width_ = 6, 6
    frame_0_ = np.linspace(
        0.2, 0.4, height_ * width_, dtype=np.float32).reshape(
            height_, width_)
    frame_1_ = frame_0_ + np.float32(1.0 / 512.0)
    assert cv2.imwrite(str(input_folder_ / "00000000.tiff"), frame_0_)
    assert cv2.imwrite(str(input_folder_ / "00000001.tiff"), frame_1_)

    args_ = _Build_v2e_args([
        "--input", str(input_folder_),
        "--input_frame_rate", "30",
        "--output_folder", str(output_folder_),
        "--unique_output_folder", "false",
        "--overwrite",
        "--disable_slomo",
        "--skip_video_output",
        "--no_preview",
        "--output_width", str(width_),
        "--output_height", str(height_),
        "--hdr",
        "--ddd_output",
        "--dvs_h5", "events.h5",
        "--dvs_emulator_seed", "123",
        "--pos_thres", "0.001",
        "--neg_thres", "0.001",
        "--sigma_thres", "0",
        "--cutoff_hz", "0",
        "--leak_rate_hz", "0",
        "--shot_noise_rate_hz", "0",
        "--refractory_period", "0",
    ])

    monkeypatch.setattr(
        v2e_module_, "Gooey", lambda *args_, **kwargs_: (lambda: None),
        raising=False)
    monkeypatch.setattr(
        v2e_module_, "get_args", lambda: (args_, [], "v2e HDR test"))
    monkeypatch.setattr(
        v2e_module_, "inputVideoFileDialog", lambda: str(input_folder_))
    monkeypatch.setattr(
        v2e_module_.desktop, "open", lambda *args_, **kwargs_: None)

    v2e_module_.main()

    with h5py_.File(output_folder_ / "events.h5", "r") as h5_file_:
        cli_frames_ = h5_file_["frame"][:]
        cli_frame_times_ = h5_file_["frame_ts"][:].astype(np.float64) / 1.0e6
        cli_events_ = h5_file_["events"][:]

    np.testing.assert_allclose(cli_frames_[0], frame_0_, atol=1.0e-7)
    np.testing.assert_allclose(cli_frames_[1], frame_1_, atol=1.0e-7)

    direct_emulator_ = emulator_module_.EventEmulator(
        pos_thres=0.001,
        neg_thres=0.001,
        sigma_thres=0.0,
        cutoff_hz=0.0,
        leak_rate_hz=0.0,
        shot_noise_rate_hz=0.0,
        refractory_period_s=0.0,
        seed=123,
        output_folder=None,
        output_width=width_,
        output_height=height_,
        dvs_h5=None,
        device=str(v2e_module_.torch_device),
        hdr=True,
        hdr_disable_prepro=False,
    )
    try:
        direct_emulator_.generate_events(
            frame_0_, float(cli_frame_times_[0]))
        direct_events_ = direct_emulator_.generate_events(
            frame_1_, float(cli_frame_times_[1]))
    finally:
        direct_emulator_.cleanup()

    assert direct_events_ is not None
    expected_events_ = direct_events_.copy()
    expected_events_[:, 0] *= 1.0e6
    expected_events_[expected_events_[:, 3] == -1, 3] = 0
    np.testing.assert_array_equal(
        cli_events_, expected_events_.astype(np.uint32))


def test_frames_repr_handles_none_transform():
    pytest.importorskip("torch")
    dataloader_module = importlib.import_module("v2ecore.dataloader")
    Frames = dataloader_module.Frames

    dataset = Frames(array=np.zeros((2, 32, 32), dtype=np.uint8), transform=None)
    representation = repr(dataset)
    assert "Transforms (if any): None" in representation


def test_frames_directory_repr_handles_none_transform(tmp_path: Path):
    pytest.importorskip("torch")
    dataloader_module = importlib.import_module("v2ecore.dataloader")
    FramesDirectory = dataloader_module.FramesDirectory

    frame_folder = tmp_path / "npy_frames"
    frame_folder.mkdir()
    np.save(frame_folder / "00000000.npy", np.zeros((32, 32), dtype=np.uint8))
    np.save(frame_folder / "00000001.npy", np.ones((32, 32), dtype=np.uint8))

    dataset = FramesDirectory(folder_path=str(frame_folder), ori_dim=(32, 32), transform=None)
    representation = repr(dataset)
    assert "Transforms (if any): None" in representation


def test_read_aedat_txt_events_uses_modern_pandas_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    pd = pytest.importorskip("pandas")
    events_text_path = tmp_path / "events.txt"
    events_text_path.write_text(
        "#!events.txt\n"
        "0.000000 10 20 1\n"
        "0.100000 12 22 0\n",
        encoding="utf-8",
    )

    captured_kwargs: dict[str, object] = {}
    real_read_table = pd.read_table

    def Wrapped_read_table(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return real_read_table(*args, **kwargs)

    monkeypatch.setattr(pd, "read_table", Wrapped_read_table)
    parsed_events = read_aedat_txt_events(str(events_text_path))

    assert captured_kwargs.get("on_bad_lines") == "warn"
    assert parsed_events.shape == (2, 4)


def test_crop_validation_rejects_invalid_crop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    v2e_module = _Import_local_v2e_module()

    class QuitCalled(RuntimeError):
        pass

    def Raise_quit(code=0):
        raise QuitCalled(code)

    input_folder = tmp_path / "input_frames_invalid_crop"
    input_folder.mkdir()
    _Write_test_image(input_folder / "00000000.png", pixel_value=10)
    _Write_test_image(input_folder / "00000001.png", pixel_value=200)

    output_folder = tmp_path / "output_invalid_crop"
    args = _Build_v2e_args([
        "--input", str(input_folder),
        "--input_frame_rate", "30",
        "--output_folder", str(output_folder),
        "--unique_output_folder", "false",
        "--overwrite",
        "--disable_slomo",
        "--skip_video_output",
        "--no_preview",
        "--output_width", "6",
        "--output_height", "6",
        "--crop", "4,3,0,0",
    ])

    monkeypatch.setattr(v2e_module, "Gooey", lambda *args, **kwargs: (lambda: None), raising=False)
    monkeypatch.setattr(v2e_module, "get_args", lambda: (args, [], "v2e test"))
    monkeypatch.setattr(v2e_module, "inputVideoFileDialog", lambda: str(input_folder))
    monkeypatch.setattr(v2e_module.desktop, "open", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(v2e_module, "v2e_quit", Raise_quit)

    with pytest.raises(QuitCalled):
        v2e_module.main()


def test_crop_validation_accepts_valid_crop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    v2e_module = _Import_local_v2e_module()

    class QuitCalled(RuntimeError):
        pass

    def Raise_quit(code=0):
        raise QuitCalled(code)

    input_folder = tmp_path / "input_frames_valid_crop"
    input_folder.mkdir()
    _Write_test_image(input_folder / "00000000.png", pixel_value=10)
    _Write_test_image(input_folder / "00000001.png", pixel_value=200)

    output_folder = tmp_path / "output_valid_crop"
    args = _Build_v2e_args([
        "--input", str(input_folder),
        "--input_frame_rate", "30",
        "--output_folder", str(output_folder),
        "--unique_output_folder", "false",
        "--overwrite",
        "--disable_slomo",
        "--skip_video_output",
        "--no_preview",
        "--output_width", "6",
        "--output_height", "6",
        "--crop", "1,1,0,0",
    ])

    monkeypatch.setattr(v2e_module, "Gooey", lambda *args, **kwargs: (lambda: None), raising=False)
    monkeypatch.setattr(v2e_module, "get_args", lambda: (args, [], "v2e test"))
    monkeypatch.setattr(v2e_module, "inputVideoFileDialog", lambda: str(input_folder))
    monkeypatch.setattr(v2e_module.desktop, "open", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(v2e_module, "v2e_quit", Raise_quit)

    v2e_module.main()
    assert output_folder.exists()


def test_main_rejects_output_in_place_for_synthetic_input(monkeypatch: pytest.MonkeyPatch):
    v2e_module = _Import_local_v2e_module()

    class QuitCalled(RuntimeError):
        pass

    def Raise_quit(code=0):
        raise QuitCalled(code)

    args = _Build_v2e_args([
        "--synthetic_input", "scripts.particles",
        "--output_in_place", "true",
    ])

    monkeypatch.setattr(v2e_module, "Gooey", lambda *args, **kwargs: (lambda: None), raising=False)
    monkeypatch.setattr(v2e_module, "get_args", lambda: (args, [], "v2e test"))
    monkeypatch.setattr(v2e_module, "v2e_quit", Raise_quit)

    with pytest.raises(QuitCalled):
        v2e_module.main()


def _Build_fake_dv_module():
    class FakeEventStore:
        def __init__(self):
            self.events = []

        def push_back(self, timestamp_us: int, x: int, y: int, polarity: int):
            self.events.append((timestamp_us, x, y, polarity))

    class FakeMonoCameraWriter:
        class EventOnlyConfig:
            def __init__(self, camera_name: str, resolution: tuple[int, int]):
                self.camera_name = camera_name
                self.resolution = resolution

        def __init__(self, filepath: str, config):
            self.filepath = filepath
            self.config = config
            self.written_batches = []

        def writeEvents(self, event_store: FakeEventStore):
            self.written_batches.append(list(event_store.events))

    class FakeDvModule:
        EventStore = FakeEventStore

        class io:
            MonoCameraWriter = FakeMonoCameraWriter

    return FakeDvModule


def test_aedat4_writer_uses_requested_resolution(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("dv_processing")
    aedat4_module = importlib.import_module("v2ecore.output.aedat4_output")
    fake_dv_module = _Build_fake_dv_module()
    monkeypatch.setattr(aedat4_module, "dv", fake_dv_module)

    writer = aedat4_module.AEDat4Output("dummy.aedat4", output_width=123, output_height=45)
    assert writer.writer.config.resolution == (123, 45)
    writer.close()


def test_aedat4_writer_uses_requested_camera_name(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("dv_processing")
    aedat4_module = importlib.import_module("v2ecore.output.aedat4_output")
    fake_dv_module = _Build_fake_dv_module()
    monkeypatch.setattr(aedat4_module, "dv", fake_dv_module)

    writer = aedat4_module.AEDat4Output(
        "dummy.aedat4", output_width=123, output_height=45, camera_name="DVXplorer")
    assert writer.writer.config.camera_name == "DVXplorer"
    writer.close()


def test_aedat4_writer_flushes_each_append_call(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("dv_processing")
    aedat4_module = importlib.import_module("v2ecore.output.aedat4_output")
    fake_dv_module = _Build_fake_dv_module()
    monkeypatch.setattr(aedat4_module, "dv", fake_dv_module)

    writer = aedat4_module.AEDat4Output("dummy.aedat4", output_width=16, output_height=12)
    first_batch = np.array([[0.001, 1, 2, 1], [0.002, 3, 4, -1]], dtype=np.float32)
    second_batch = np.array([[0.003, 5, 6, 1]], dtype=np.float32)

    writer.appendEvents(first_batch)
    writer.appendEvents(second_batch)

    assert len(writer.writer.written_batches) == 2
    assert len(writer.writer.written_batches[0]) == 2
    assert len(writer.writer.written_batches[1]) == 1
    writer.close()


def test_resolve_dvs_emulator_seed_uses_explicit_seed():
    v2e_module = _Import_local_v2e_module()
    effective_seed, auto_generated = v2e_module.resolve_dvs_emulator_seed(1234)
    assert effective_seed == 1234
    assert auto_generated is False


def test_resolve_dvs_emulator_seed_auto_generates_when_zero(monkeypatch: pytest.MonkeyPatch):
    v2e_module = _Import_local_v2e_module()
    generated = iter([111, 222])
    monkeypatch.setattr(
        v2e_module.np.random, "randint", lambda _low, _high: next(generated))

    first_seed, first_auto = v2e_module.resolve_dvs_emulator_seed(0)
    second_seed, second_auto = v2e_module.resolve_dvs_emulator_seed(0)

    assert first_auto is True
    assert second_auto is True
    assert first_seed == 111
    assert second_seed == 222
    assert first_seed != second_seed


def test_main_resolves_auto_generated_seed_once(monkeypatch: pytest.MonkeyPatch,
                                                tmp_path: Path,
                                                caplog: pytest.LogCaptureFixture) -> None:
    v2e_module = _Import_local_v2e_module()

    class StopAfterSeed(RuntimeError):
        pass

    captured_seed: dict[str, int] = {}

    class FakeEventEmulator:
        def __init__(self, *_args: object, **kwargs_: object) -> None:
            captured_seed["value"] = int(kwargs_["seed"])
            raise StopAfterSeed("seed captured")

    input_folder = tmp_path / "input_frames_seed_test"
    input_folder.mkdir()
    _Write_test_image(input_folder / "00000000.png", pixel_value=10)
    _Write_test_image(input_folder / "00000001.png", pixel_value=20)

    output_folder = tmp_path / "output_seed_test"
    args = _Build_v2e_args([
        "--input", str(input_folder),
        "--input_frame_rate", "30",
        "--output_folder", str(output_folder),
        "--unique_output_folder", "false",
        "--overwrite",
        "--disable_slomo",
        "--skip_video_output",
        "--no_preview",
        "--output_width", "6",
        "--output_height", "6",
        "--dvs_emulator_seed", "0",
    ])

    seed_draw_mock_ = Mock(side_effect=[54321, 99999])
    monkeypatch.setattr(v2e_module.np.random, "randint", seed_draw_mock_)
    monkeypatch.setattr(v2e_module, "EventEmulator", FakeEventEmulator)
    monkeypatch.setattr(v2e_module, "Gooey", lambda *args, **kwargs: (lambda: None), raising=False)
    monkeypatch.setattr(v2e_module, "get_args", lambda: (args, [], "v2e test"))
    monkeypatch.setattr(v2e_module, "inputVideoFileDialog", lambda: str(input_folder))
    monkeypatch.setattr(v2e_module.desktop, "open", lambda *_args, **_kwargs: None)

    with caplog.at_level(logging.INFO):
        with pytest.raises(StopAfterSeed):
            v2e_module.main()

    assert captured_seed["value"] == 54321
    assert "Using DVS emulator seed: 54321 (auto-generated)" in caplog.text
    seed_draw_mock_.assert_called_once_with(1, 2**31)


def test_main_passes_iebcs_and_v2ce_flags_to_emulator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    v2e_module = _Import_local_v2e_module()

    class StopAfterCapture(RuntimeError):
        pass

    captured_kwargs: dict[str, object] = {}

    class FakeEventEmulator:
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            raise StopAfterCapture("kwargs captured")

    input_folder = tmp_path / "input_frames_feature_flags"
    input_folder.mkdir()
    _Write_test_image(input_folder / "00000000.png", pixel_value=10)
    _Write_test_image(input_folder / "00000001.png", pixel_value=220)

    output_folder = tmp_path / "output_feature_flags"
    pos_file = tmp_path / "pos.npy"
    neg_file = tmp_path / "neg.npy"
    np.save(pos_file, np.array([[0.0, 1.0]], dtype=np.float32))
    np.save(neg_file, np.array([[0.0, 1.0]], dtype=np.float32))
    args = _Build_v2e_args([
        "--input", str(input_folder),
        "--input_frame_rate", "30",
        "--output_folder", str(output_folder),
        "--unique_output_folder", "false",
        "--overwrite",
        "--disable_slomo",
        "--skip_video_output",
        "--no_preview",
        "--output_width", "6",
        "--output_height", "6",
        "--dvs_emulator_seed", "123",
        "--iebcs_latency_jitter_model", "true",
        "--iebcs_latency_mean_us", "210",
        "--iebcs_latency_jitter_us", "11",
        "--iebcs_resample_thresholds_on_event", "true",
        "--iebcs_contrast_latency_model", "true",
        "--iebcs_latency_tau_us", "320",
        "--iebcs_latency_clamp_us", "9000",
        "--iebcs_latency_slope_jitter", "false",
        "--iebcs_hist_noise_model", "true",
        "--iebcs_noise_source", "files",
        "--iebcs_noise_pos_path", str(pos_file),
        "--iebcs_noise_neg_path", str(neg_file),
        "--iebcs_refractory_state_coupling", "true",
        "--iebcs_refractory_us", "700",
        "--v2ce_nonuniform_burst_timestamps", "true",
        "--v2ce_burst_timestamps_mode", "slope",
        "--strict_model_validity",
    ])

    monkeypatch.setattr(v2e_module, "EventEmulator", FakeEventEmulator)
    monkeypatch.setattr(v2e_module, "Gooey", lambda *args, **kwargs: (lambda: None), raising=False)
    monkeypatch.setattr(v2e_module, "get_args", lambda: (args, [], "v2e test"))
    monkeypatch.setattr(v2e_module, "inputVideoFileDialog", lambda: str(input_folder))
    monkeypatch.setattr(v2e_module.desktop, "open", lambda *_args, **_kwargs: None)

    with pytest.raises(StopAfterCapture):
        v2e_module.main()

    assert captured_kwargs["seed"] == 123
    assert captured_kwargs["iebcs_latency_jitter_model"] is True
    assert captured_kwargs["iebcs_latency_mean_us"] == 210.0
    assert captured_kwargs["iebcs_latency_jitter_us"] == 11.0
    assert captured_kwargs["iebcs_resample_thresholds_on_event"] is True
    assert captured_kwargs["iebcs_contrast_latency_model"] is True
    assert captured_kwargs["iebcs_latency_tau_us"] == 320.0
    assert captured_kwargs["iebcs_latency_clamp_us"] == 9000.0
    assert captured_kwargs["iebcs_latency_slope_jitter"] is False
    assert captured_kwargs["iebcs_hist_noise_model"] is True
    assert captured_kwargs["iebcs_hist_noise_pos_path"] == str(pos_file)
    assert captured_kwargs["iebcs_hist_noise_neg_path"] == str(neg_file)
    assert captured_kwargs["iebcs_refractory_state_coupling"] is True
    assert captured_kwargs["iebcs_refractory_us"] == 700.0
    assert captured_kwargs["v2ce_nonuniform_burst_timestamps"] is True
    assert captured_kwargs["v2ce_burst_timestamps_mode"] == "slope"
    assert captured_kwargs["strict_model_validity"] is True


def test_set_output_dimension_supports_dvxplorer_preset():
    width, height = set_output_dimension(
        output_width=None,
        output_height=None,
        dvs128=False,
        dvs240=False,
        dvs346=False,
        dvs640=False,
        dvs1024=False,
        dvxplorer=True,
        logger=logging.getLogger(__name__),
    )
    assert (width, height) == (640, 480)


def test_main_scales_large_resolution_even_when_slomo_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    v2e_module = _Import_local_v2e_module()

    class StopAfterDimensions(RuntimeError):
        pass

    captured_dimensions: dict[str, int] = {}

    class FakeEventEmulator:
        def __init__(self, *args, **kwargs):
            captured_dimensions["output_width"] = kwargs["output_width"]
            captured_dimensions["output_height"] = kwargs["output_height"]
            raise StopAfterDimensions("dimensions captured")

    input_folder = tmp_path / "input_frames_large_resolution"
    input_folder.mkdir()
    large_image = np.full((1536, 2048, 3), 100, dtype=np.uint8)
    assert cv2.imwrite(str(input_folder / "00000000.png"), large_image)
    assert cv2.imwrite(str(input_folder / "00000001.png"), large_image)

    output_folder = tmp_path / "output_large_resolution"
    args = _Build_v2e_args([
        "--input", str(input_folder),
        "--input_frame_rate", "1",
        "--output_folder", str(output_folder),
        "--unique_output_folder", "false",
        "--overwrite",
        "--disable_slomo",
        "--skip_video_output",
        "--no_preview",
    ])

    monkeypatch.setattr(v2e_module, "EventEmulator", FakeEventEmulator)
    monkeypatch.setattr(v2e_module, "Gooey", lambda *args, **kwargs: (lambda: None), raising=False)
    monkeypatch.setattr(v2e_module, "get_args", lambda: (args, [], "v2e test"))
    monkeypatch.setattr(v2e_module, "inputVideoFileDialog", lambda: str(input_folder))
    monkeypatch.setattr(v2e_module.desktop, "open", lambda *_args, **_kwargs: None)

    with pytest.raises(StopAfterDimensions):
        v2e_module.main()

    assert captured_dimensions["output_width"] == 1024
    assert captured_dimensions["output_height"] == 768


def test_hist_noise_presets_resolve_correctly():
    v2e_module = _Import_local_v2e_module()
    pos, neg = v2e_module.resolve_iebcs_noise_paths(
        noise_source="preset",
        noise_preset="161lux",
        noise_pos_path=None,
        noise_neg_path=None,
    )
    assert pos.endswith("input/iebcs_noise/noise_pos_161lux.npy")
    assert neg.endswith("input/iebcs_noise/noise_neg_161lux.npy")


def test_cli_rejects_invalid_hist_noise_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    v2e_module = _Import_local_v2e_module()

    class QuitCalled(RuntimeError):
        pass

    def Raise_quit(code=0):
        raise QuitCalled(code)

    input_folder = tmp_path / "input_frames_invalid_hist"
    input_folder.mkdir()
    _Write_test_image(input_folder / "00000000.png", pixel_value=20)
    _Write_test_image(input_folder / "00000001.png", pixel_value=25)

    output_folder = tmp_path / "output_invalid_hist"
    args = _Build_v2e_args([
        "--input", str(input_folder),
        "--input_frame_rate", "30",
        "--output_folder", str(output_folder),
        "--unique_output_folder", "false",
        "--overwrite",
        "--disable_slomo",
        "--skip_video_output",
        "--no_preview",
        "--output_width", "6",
        "--output_height", "6",
        "--iebcs_hist_noise_model", "true",
        "--iebcs_noise_source", "files",
    ])

    monkeypatch.setattr(v2e_module, "Gooey", lambda *args, **kwargs: (lambda: None), raising=False)
    monkeypatch.setattr(v2e_module, "get_args", lambda: (args, [], "v2e test"))
    monkeypatch.setattr(v2e_module, "v2e_quit", Raise_quit)

    with pytest.raises(QuitCalled):
        v2e_module.main()


def test_cli_rejects_missing_preset_hist_noise_files(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture):
    v2e_module = _Import_local_v2e_module()

    class QuitCalled(RuntimeError):
        pass

    def Raise_quit(code=0):
        raise QuitCalled(code)

    input_folder = tmp_path / "input_frames_missing_preset_hist"
    input_folder.mkdir()
    _Write_test_image(input_folder / "00000000.png", pixel_value=20)
    _Write_test_image(input_folder / "00000001.png", pixel_value=25)

    output_folder = tmp_path / "output_missing_preset_hist"
    args = _Build_v2e_args([
        "--input", str(input_folder),
        "--input_frame_rate", "30",
        "--output_folder", str(output_folder),
        "--unique_output_folder", "false",
        "--overwrite",
        "--disable_slomo",
        "--skip_video_output",
        "--no_preview",
        "--output_width", "6",
        "--output_height", "6",
        "--iebcs_hist_noise_model", "true",
        "--iebcs_noise_source", "preset",
        "--iebcs_noise_preset", "161lux",
    ])

    monkeypatch.setattr(v2e_module, "Gooey", lambda *args, **kwargs: (lambda: None), raising=False)
    monkeypatch.setattr(v2e_module, "get_args", lambda: (args, [], "v2e test"))
    monkeypatch.setattr(v2e_module, "v2e_quit", Raise_quit)

    with pytest.raises(QuitCalled):
        v2e_module.main()
    assert "IEBCS histogram noise file(s) not found" in caplog.text


def test_cli_rejects_negative_latency_tau_or_refractory_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    v2e_module = _Import_local_v2e_module()

    class QuitCalled(RuntimeError):
        pass

    def Raise_quit(code=0):
        raise QuitCalled(code)

    input_folder = tmp_path / "input_frames_negative_checks"
    input_folder.mkdir()
    _Write_test_image(input_folder / "00000000.png", pixel_value=20)
    _Write_test_image(input_folder / "00000001.png", pixel_value=25)

    output_folder = tmp_path / "output_negative_checks"
    args = _Build_v2e_args([
        "--input", str(input_folder),
        "--input_frame_rate", "30",
        "--output_folder", str(output_folder),
        "--unique_output_folder", "false",
        "--overwrite",
        "--disable_slomo",
        "--skip_video_output",
        "--no_preview",
        "--output_width", "6",
        "--output_height", "6",
        "--iebcs_latency_tau_us", "-1",
    ])
    monkeypatch.setattr(v2e_module, "Gooey", lambda *args, **kwargs: (lambda: None), raising=False)
    monkeypatch.setattr(v2e_module, "get_args", lambda: (args, [], "v2e test"))
    monkeypatch.setattr(v2e_module, "v2e_quit", Raise_quit)
    with pytest.raises(QuitCalled):
        v2e_module.main()

    args2 = _Build_v2e_args([
        "--input", str(input_folder),
        "--input_frame_rate", "30",
        "--output_folder", str(output_folder),
        "--unique_output_folder", "false",
        "--overwrite",
        "--disable_slomo",
        "--skip_video_output",
        "--no_preview",
        "--output_width", "6",
        "--output_height", "6",
        "--iebcs_refractory_us", "-1",
    ])
    monkeypatch.setattr(v2e_module, "get_args", lambda: (args2, [], "v2e test"))
    with pytest.raises(QuitCalled):
        v2e_module.main()

import argparse
import importlib
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("easygui")

from v2ecore.v2e_utils import ImageFolderReader
from v2ecore.v2e_utils import read_aedat_txt_events
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


def test_set_output_folder_rejects_output_in_place_without_input_path():
    with pytest.raises(ValueError, match="output_in_place=True requires a real input file or input folder path"):
        set_output_folder(
            output_folder="v2e-output",
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
    v2e_module = pytest.importorskip("v2e")

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
    monkeypatch.setattr(v2e_module.desktop, "open", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(v2e_module, "v2e_quit", Raise_quit)

    with pytest.raises(QuitCalled):
        v2e_module.main()


def test_crop_validation_accepts_valid_crop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    v2e_module = pytest.importorskip("v2e")

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
    monkeypatch.setattr(v2e_module.desktop, "open", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(v2e_module, "v2e_quit", Raise_quit)

    v2e_module.main()
    assert output_folder.exists()


def test_main_rejects_output_in_place_for_synthetic_input(monkeypatch: pytest.MonkeyPatch):
    v2e_module = pytest.importorskip("v2e")

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

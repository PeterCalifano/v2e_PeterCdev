from pathlib import Path


def test_quick_script_uses_auto_timestamp_resolution_flag():
    script = Path("frames2events_quick.sh").read_text()
    assert "--auto_timestamp_resolution" in script
    assert "--auto_timestamp " not in script

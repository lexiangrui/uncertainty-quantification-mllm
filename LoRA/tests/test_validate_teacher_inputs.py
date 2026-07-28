from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_teacher_inputs.py"
SPEC = importlib.util.spec_from_file_location("validate_teacher_inputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_is_jpeg_checks_file_content_not_extension(tmp_path) -> None:
    jpeg = tmp_path / "real.jpg"
    jpeg.write_bytes(b"\xff\xd8\xffpayload")
    fake = tmp_path / "fake.jpg"
    fake.write_bytes(b"RIFFxxxxWEBP")
    assert MODULE.is_jpeg(jpeg)
    assert not MODULE.is_jpeg(fake)

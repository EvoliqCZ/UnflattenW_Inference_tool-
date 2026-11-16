import zipfile
from io import BytesIO

import pytest

from unflatten.renderer.opc import PackagePart, build_package


def test_build_package_entry_order_preserved():
    parts = [PackagePart("b.xml", b"<b/>"), PackagePart("a.xml", b"<a/>")]
    data = build_package(parts)
    with zipfile.ZipFile(BytesIO(data)) as zf:
        assert zf.namelist() == ["b.xml", "a.xml"]


def test_build_package_fixed_timestamp():
    parts = [PackagePart("a.xml", b"<a/>")]
    data = build_package(parts)
    with zipfile.ZipFile(BytesIO(data)) as zf:
        info = zf.getinfo("a.xml")
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert info.compress_type == zipfile.ZIP_DEFLATED


def test_build_package_duplicate_names_rejected():
    parts = [PackagePart("a.xml", b"<a/>"), PackagePart("a.xml", b"<a2/>")]
    with pytest.raises(ValueError):
        build_package(parts)


def test_build_package_deterministic_bytes():
    parts = [PackagePart("a.xml", b"<a/>"), PackagePart("b.xml", b"<b/>")]
    assert build_package(parts) == build_package(parts)


def test_build_package_roundtrip_content():
    parts = [PackagePart("a.xml", b"<a>hello</a>")]
    data = build_package(parts)
    with zipfile.ZipFile(BytesIO(data)) as zf:
        assert zf.read("a.xml") == b"<a>hello</a>"

import hashlib
from pathlib import Path

import pytest

from unflatten.renderer.resources import DirResourceProvider, resolve_resource

ASSETS_DIR = Path(__file__).parent.parent / "data" / "assets"
_PNG_BYTES = (ASSETS_DIR / "tiny_2x2.png").read_bytes()
_PNG_SHA256 = hashlib.sha256(_PNG_BYTES).hexdigest()


def _entry(**overrides) -> dict:
    base = {
        "id": "res_1",
        "kind": "image",
        "mime_type": "image/png",
        "sha256": _PNG_SHA256,
        "path": "tiny_2x2.png",
        "pixel_width": 2,
        "pixel_height": 2,
    }
    base.update(overrides)
    return base


def test_resolve_resource_success():
    provider = DirResourceProvider(ASSETS_DIR)
    resolved, issue = resolve_resource(_entry(), provider, "/resources/0")
    assert issue is None
    assert resolved.sha256 == _PNG_SHA256
    assert resolved.extension == "png"
    assert resolved.data == _PNG_BYTES
    assert resolved.media_part_name == f"word/media/{_PNG_SHA256[:16]}.png"


def test_resolve_resource_hash_mismatch():
    provider = DirResourceProvider(ASSETS_DIR)
    entry = _entry(sha256="0" * 64)
    resolved, issue = resolve_resource(entry, provider, "/resources/0")
    assert resolved is None
    assert issue.code == "RESOURCE_HASH_MISMATCH"
    assert issue.severity == "error"


@pytest.mark.parametrize("bad_path", ["../evil.png", "/etc/passwd", "images/../../evil.png"])
def test_resolve_resource_rejects_path_traversal(bad_path):
    provider = DirResourceProvider(ASSETS_DIR)
    entry = _entry(path=bad_path)
    resolved, issue = resolve_resource(entry, provider, "/resources/0")
    assert resolved is None
    assert issue.code == "RESOURCE_PATH_TRAVERSAL"


def test_resolve_resource_rejects_unsupported_mime():
    provider = DirResourceProvider(ASSETS_DIR)
    entry = _entry(mime_type="image/svg+xml")
    resolved, issue = resolve_resource(entry, provider, "/resources/0")
    assert resolved is None
    assert issue.code == "RESOURCE_MIME_UNSUPPORTED"


def test_resolve_resource_not_found():
    provider = DirResourceProvider(ASSETS_DIR)
    entry = _entry(path="does_not_exist.png")
    resolved, issue = resolve_resource(entry, provider, "/resources/0")
    assert resolved is None
    assert issue.code == "RESOURCE_NOT_FOUND"


def test_resolve_resource_too_large():
    provider = DirResourceProvider(ASSETS_DIR)
    resolved, issue = resolve_resource(_entry(), provider, "/resources/0", max_bytes=10)
    assert resolved is None
    assert issue.code == "RESOURCE_TOO_LARGE"


def test_dir_resource_provider_rejects_symlink_escaping_root(tmp_path):
    root = tmp_path / "resource_root"
    root.mkdir()
    outside = tmp_path / "outside_secret.png"
    outside.write_bytes(_PNG_BYTES)
    (root / "innocent_looking.png").symlink_to(outside)

    provider = DirResourceProvider(root)
    with pytest.raises(Exception, match="escapes resource root"):
        provider.open("innocent_looking.png")


def test_resolve_resource_rejects_symlink_escaping_root_via_resolve_resource(tmp_path):
    root = tmp_path / "resource_root"
    root.mkdir()
    outside = tmp_path / "outside_secret.png"
    outside.write_bytes(_PNG_BYTES)
    (root / "innocent_looking.png").symlink_to(outside)

    provider = DirResourceProvider(root)
    entry = _entry(path="innocent_looking.png")
    resolved, issue = resolve_resource(entry, provider, "/resources/0")
    assert resolved is None
    assert issue.code == "RESOURCE_PATH_TRAVERSAL"


def test_dir_resource_provider_allows_symlink_that_stays_inside_root(tmp_path):
    root = tmp_path / "resource_root"
    root.mkdir()
    real = root / "real.png"
    real.write_bytes(_PNG_BYTES)
    (root / "alias.png").symlink_to(real)

    provider = DirResourceProvider(root)
    assert provider.open("alias.png") == _PNG_BYTES

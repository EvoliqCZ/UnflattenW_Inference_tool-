from __future__ import annotations

import importlib.util
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from unflatten.service.pdf_sandbox import PdfAdmissionReport, PdfSandbox, PdfSandboxFailure
from unflatten.service.policy import load_runtime_policy
from unflatten.service.worker import DiagnosticWorker, WorkerFailure


def _scanner(monkeypatch, pages):
    path = Path(__file__).parents[2] / 'deploy/pdf-sandbox/scanner.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('geometry_scanner', path)
    assert spec and spec.loader
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    policy = load_runtime_policy()
    for key, value in {'RASTER_DPI': 300, 'MAX_PAGE_WIDTH_POINTS': 1440,
                       'MAX_PAGE_HEIGHT_POINTS': 1440, 'MAX_RASTER_WIDTH_PIXELS': 6000,
                       'MAX_RASTER_HEIGHT_PIXELS': 6000, 'MAX_RASTER_PIXELS': 36000000,
                       'MAX_PDF_PAGES': 100}.items():
        monkeypatch.setenv(key, str(value))

    def run(command):
        if command[0] == 'pdfinfo':
            selected = pages[:int(command[command.index('-l') + 1])]
            info = f'Encrypted: no\nPages: {len(pages)}\n' + ''.join(
                f'Page {i} size: {w} x {h} pts\nPage {i} rot: {r}\n'
                for i, (w, h, r) in enumerate(selected, 1))
            return subprocess.CompletedProcess(command, 0, info.encode())
        return subprocess.CompletedProcess(command, 0, json.dumps({'qpdf': []}).encode())

    monkeypatch.setattr(scanner, '_run', run)
    return scanner, policy


@pytest.mark.parametrize(('pages', 'expected'), [
    ([(72, 144, 0), (144, 72, 0)], [(300, 600), (600, 300)]),
    ([(72, 144, 90)], [(600, 300)]),
    ([(72, 144, 270)], [(600, 300)]),
])
def test_inspect_pdf_each_visual_page_geometry_matches_rotation(monkeypatch, pages, expected):
    scanner, _ = _scanner(monkeypatch, pages)
    report = scanner.inspect_pdf(Path('source.pdf'))
    assert report['status'] == 'accepted'
    assert [(p['raster_width_pixels'], p['raster_height_pixels'])
            for p in report['page_geometries']] == expected


def test_inspect_pdf_oversized_later_page_is_refused(monkeypatch):
    scanner, _ = _scanner(monkeypatch, [(72, 144, 0), (1441, 72, 0)])
    report = scanner.inspect_pdf(Path('source.pdf'))
    assert report['reason'] == 'PDF_PAGE_DIMENSIONS_EXCEEDED'


@pytest.mark.parametrize('wrong_geometry', [False, True])
def test_raster_selected_page_validates_its_geometry(monkeypatch, tmp_path, wrong_geometry):
    scanner, policy = _scanner(monkeypatch, [(72, 144, 0), (144, 72, 0)])
    report = PdfAdmissionReport.from_mapping(scanner.inspect_pdf(Path('source.pdf')))
    image = tmp_path / 'fake.png'
    Image.new('RGB', (300, 600) if wrong_geometry else (600, 300)).save(image)
    sandbox = PdfSandbox(policy=policy, image='unflatten/pdf-sandbox:local',
                         allow_rootful=True, runtime='docker')
    monkeypatch.setattr(sandbox, '_invoke', lambda *a, **kw: image.read_bytes())
    source = tmp_path / 'source.pdf'
    source.write_bytes(b'%PDF-1.7\n')
    output = tmp_path / 'page.png'
    if wrong_geometry:
        with pytest.raises(PdfSandboxFailure) as failure:
            sandbox.raster(source, report, output, page_index=1)
        assert failure.value.code == 'RASTER_OUTPUT_INVALID'
    else:
        sandbox.raster(source, report, output, page_index=1)
        assert output.read_bytes() == image.read_bytes()


def test_load_rgb_second_page_uses_its_own_geometry(monkeypatch, tmp_path):
    scanner, policy = _scanner(monkeypatch, [(72, 144, 0), (144, 72, 0)])
    report = PdfAdmissionReport.from_mapping(scanner.inspect_pdf(Path('source.pdf')))
    image = tmp_path / 'page.png'
    Image.new('RGB', (600, 300)).save(image)
    worker = DiagnosticWorker.__new__(DiagnosticWorker)
    worker._policy = policy
    width, height, _, _ = worker._load_rgb(image, report, page_index=1)
    assert (width, height) == (600, 300)
    Image.new('RGB', (300, 600)).save(image)
    with pytest.raises(WorkerFailure):
        worker._load_rgb(image, report, page_index=1)


@pytest.mark.parametrize('mutation', ['missing_page', 'extra_key', 'wrong_pixel_count'])
def test_report_per_page_geometry_invalid_report_is_refused(monkeypatch, mutation):
    scanner, _ = _scanner(monkeypatch, [(72, 144, 0), (144, 72, 0)])
    report = scanner.inspect_pdf(Path('source.pdf'))
    if mutation == 'missing_page':
        report['page_geometries'].pop()
    elif mutation == 'extra_key':
        report['page_geometries'][1]['metadata'] = 'forbidden'
    else:
        report['page_geometries'][1]['raster_pixels'] += 1
    with pytest.raises(PdfSandboxFailure) as failure:
        PdfAdmissionReport.from_mapping(report)
    assert failure.value.code == 'PDF_MALFORMED'


def test_entrypoint_raster_renders_visible_cropbox(monkeypatch, tmp_path):
    if not shutil.which('pdftoppm'):
        pytest.skip('Poppler is required for this raster geometry test')
    pymupdf = pytest.importorskip('pymupdf')
    source = tmp_path / 'source.pdf'
    document = pymupdf.open()
    page = document.new_page(width=72, height=144)
    page.set_cropbox(pymupdf.Rect(0, 0, 36, 72))
    document.save(source)
    document.close()
    path = Path(__file__).parents[2] / 'deploy/pdf-sandbox/entrypoint.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('geometry_entrypoint', path)
    assert spec and spec.loader
    entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entrypoint)
    monkeypatch.setattr(entrypoint, 'inspect_pdf',
                        lambda p: {'status': 'accepted', 'page_count': 1})
    monkeypatch.setattr(entrypoint, 'Path',
                        lambda p: tmp_path / 'page' if p == '/tmp/page' else Path(p))
    monkeypatch.setattr(sys, 'argv', ['entrypoint', 'raster', str(source),
                                     '--expected-sha256', entrypoint._sha256(source)])
    stdout = io.TextIOWrapper(io.BytesIO())
    monkeypatch.setattr(sys, 'stdout', stdout)
    assert entrypoint.main() == 0
    stdout.flush()
    stdout.buffer.seek(0)
    with Image.open(stdout.buffer) as raster:
        assert raster.size == (150, 300)

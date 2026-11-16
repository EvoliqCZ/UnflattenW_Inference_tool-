# PDF admission sandbox (docker or podman)

This image is the mandatory hostile-PDF boundary for the private diagnostic service. It
contains qpdf `11.9.0-1.1ubuntu0.1` and Poppler (poppler-utils) `24.02.0-1ubuntu9.9` — the
same Poppler release line used to render the training rasters — with Python `3.12.3-0ubuntu2.1`,
all installed from a fixed `snapshot.ubuntu.com` snapshot (`20260924T000000Z`) on the
digest-pinned `ubuntu:24.04` base
(`docker.io/library/ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3`).
The host adapter (`PdfSandbox`, `src/unflatten/service/pdf_sandbox.py`) supports both
`docker` and `podman` runtimes (D8): it refuses tag-only image references and reports
`pdf_sandbox` unready unless the runtime is rootless (or `UNFLATTEN_SANDBOX_ALLOW_ROOTFUL=1`
is set for a rootful docker daemon — rootful podman is never accepted, G7) and the locally
available digest- or ID-pinned image carries the exact scanner identity label. There is no
host-tool fallback.

Build and record the resulting immutable reference:

```bash
# podman (registry-style tag, pin by digest)
podman build --pull=never -t localhost/unflatten-pdf-sandbox:1 \
  -f deploy/pdf-sandbox/Containerfile deploy/pdf-sandbox
podman image inspect --format '{{.Digest}}' localhost/unflatten-pdf-sandbox:1
# → configure UNFLATTEN_SANDBOX_IMAGE=localhost/unflatten-pdf-sandbox@sha256:<digest>

# docker (local build has no registry digest; pin by local image ID)
docker build -f deploy/pdf-sandbox/Containerfile -t unflatten/pdf-sandbox:local \
  deploy/pdf-sandbox
docker image inspect --format '{{.Id}}' unflatten/pdf-sandbox:local
# → configure UNFLATTEN_SANDBOX_IMAGE=sha256:<id>
```

A deployment must pre-pull/build that exact image; inability to verify it is a readiness
failure.

The adapter starts a fresh container with no network, no capabilities,
`no-new-privileges`, a read-only root, bounded tmpfs, PID/memory/CPU/open-file limits, and a
read-only single-file input mount. Under podman it additionally uses `--userns=keep-id` and
an explicitly UID/GID-owned tmpfs; docker has no `keep-id` userns equivalent, so the docker
branch omits it and uses `mode=1777` tmpfs instead (verified below). Both runtimes pass
`--user=<uid>:<gid>` so the sandboxed process runs as the invoking (non-root) user rather
than image-default root, which is how it can read a `0600` bind-mounted source file.
Inspection emits only the closed count/status report.
Scanner identity `v13` uses one fixed visible-raster admission profile (DEC-062).
Ordinary annotations and hyperlinks, hidden OCR text and ambiguous text visibility are
admitted; only rendered visible page pixels reach the model. Their classifier counts remain
in the closed report as diagnostics and never become text input. JavaScript and dangerous
actions, embedded attachments, encryption, failed structural checks and exceeded dimensions
remain blocked, with the narrow C2PA provenance exception below. The service policy bounds
PDF input to 100 pages.
`qpdf --check` must return zero. A `qpdf --json` warning (exit 3) is parsed only to identify
an explicit active-content, attachment or image-dimension refusal. Without such a refusal,
the warning remains `PDF_MALFORMED`, except for the exact C2PA spelling warning below.
Other nonzero JSON exits and unusable JSON remain malformed.

C2PA metadata is ignored for conversion, without parsing or verifying its contents. An
indirect Filespec with `/AFRelationship /C2PA_Manifest` must be declared by a catalog's
associated-files list or EmbeddedFiles name tree, or a FileAttachment annotation. Its `/EF`
stream must be exclusively referenced by that Filespec; ordinary, shared, undeclared,
unresolved or mixed attachments remain blocked. The observed Content Credentials writer's
`/FileSpec` spelling and nested C2PA stream dictionary are supported. For that spelling only,
JSON exit 3 is admitted when every stderr warning exactly identifies a known admitted C2PA
Filespec's type mismatch and qpdf reports success with warnings; additional or unmatched
warnings remain malformed. Active content is checked independently and always refused.
The closed report schema is `1.2`: `attachment_objects` retains all attachment carriers and
EF streams; `c2pa_metadata_objects` counts the admitted subset. The adapter rejects a
metadata count exceeding the total, and refuses any remaining ordinary attachment. No
metadata strings, manifest bytes or verification claims leave the scanner.
`page_geometries` records the visible CropBox dimensions and raster size of each page,
including its rotation; mixed page sizes are supported and checked individually.
The scanner is split into image-local `scanner*.py` modules; the `scanner.py` entrypoint and
closed report schema remain stable. Rebuild the image and replace the configured digest when
upgrading from an earlier scanner identity.
Raster mode repeats admission and source-hash verification, renders the requested page at
300 DPI, and emits only the PNG on stdout. It never invokes `pdftotext`.

The measurements and classifier descriptions below document earlier scanner revisions;
their visibility classifications remain diagnostic in v13, rather than admission refusals.

## T7 measurements (2026-09-25, this host: rootful docker only, no podman)

- **Package versions** — `poppler-utils=24.02.0-1ubuntu9.9`, `qpdf=11.9.0-1.1ubuntu0.1`,
  `python3=3.12.3-0ubuntu2.1`, resolved from `snapshot.ubuntu.com/ubuntu/20260924T000000Z`
  against `noble`/`noble-updates`/`noble-security`; identical to what today's unpinned
  `archive.ubuntu.com` mirror offers as candidate, so the snapshot is current.
- **Docker security flags** — `--tmpfs=/tmp:...,mode=1777` is accepted (docker rejects
  `uid=`/`gid=` on `--tmpfs`, confirming the D8 OPEN item; the podman branch keeps
  `uid=`/`gid=`). `--mount=type=bind,...,ro=true` is accepted. `--ulimit=cpu=<n>:<n>` is
  accepted. All three verified by a real `docker run` (see `tests/integration/test_pdf_sandbox_oci.py`)
  and by the tabular argv tests in `tests/service/test_pdf_sandbox.py`.
- **`--user=<uid>:<gid>` under rootful docker** — accepted; the sandboxed process
  successfully reads a `0600` bind-mounted source file owned by the invoking user (verified
  manually with `docker run ... --user=$(id -u):$(id -g)` against a `chmod 600` PDF).
  Rootless docker was **not** available on this host and remains unmeasured (P7 stays open
  for that specific combination); this closes the rootful case.
- **Scanner corpus (R4, qpdf 11.9 vs 11.3 / poppler 24.02 vs 22.12)** — `tests/service/test_pdf_scanner.py`
  (31 tests, pure classifier logic, no container) passes unchanged against the new image's
  scanner code. Running the containerized scanner against real single-page training PDFs
  (see raster parity below) surfaced one real behavioural difference: Poppler 24.02's
  `pdfinfo` appends a parenthesised paper-format label to the `Page 1 size:` line for
  well-known sizes (e.g. `612 x 792 pts (letter)`), which Poppler 22.12 did not emit. The
  scanner's `_page_size` regex required an exact `pts` end-of-line and silently misread every
  such page as malformed (`PDF_MALFORMED`) — i.e. **every US-Letter or similarly named page
  would have been rejected** under the new image. Fixed in `deploy/pdf-sandbox/scanner.py` by
  making the trailing `(<name>)` label optional in the regex; the object-graph/text-visibility
  classifiers themselves are unchanged. No other behavioural difference was observed.
- **Raster parity with training rasters** — 3 real single-page-1 rasters were pulled from
  `~/unflatten-data/m2a-v7/renders.jsonl` (`objects/` store: PDF + page-0 PNG pairs, sizes
  2550×3300 and 2481×3508 px, i.e. already 300 DPI). Running `pdftoppm -r 300 -singlefile -png`
  inside the new sandbox image against each source PDF produced output that is **byte-identical**
  (`cmp` and SHA-256 match) to the corresponding training-cache PNG for all 3 documents; pixel
  diff confirmed `max_abs_diff = 0` for all three. (Two of the three PDFs are multi-page or
  carry active content and are rejected by the service's own admission policy — expected and
  unrelated to raster fidelity; the raw `pdftoppm` comparison bypasses that policy gate on
  purpose to isolate the raster step.)


## Scanner identity v8 (2026-09-26)

`/A` and `/OpenAction` count as active content only when their value is (or resolves to) an action of subtype JavaScript/Launch/SubmitForm/ImportData/GoToR, has `/JS`, or cannot be resolved; a tagged PDF's structure attributes (`/A` with `/O`) and page-destination `/OpenAction` arrays no longer refuse the document. Found by the first real T11 acceptance: every LibreOffice/Word PDF was refused as `PDF_ACTIVE_CONTENT`. Rebuild the image and re-pin its digest.

## Scanner identity v9 (2026-09-26)

Text visibility is now **positional** instead of blanket-ambiguous. The first real T11
acceptance refused every LibreOffice/Word PDF as `PDF_TEXT_VISIBILITY_AMBIGUOUS`, because
v8 marked text ambiguous for any `Td`/`TD` offset, any non-zero `TJ` kerning number, any
non-identity `Tm` or `cm`, any clip (`W`/`W*`), and any second text run on the page — while a
typical page is `q 0 0.028 595.276 841.861 re W* n … q 0 0 0 rg BT 71 796.889 Td /F1 10 Tf
[<01>1<02>-1 …] TJ ET Q`, followed by table rules stroked after the cell text.

What v9 tracks, all in device space (`scanner_geometry.py`, `scanner_paint.py`):

- **CTM** — `cm` composes into the graphics state (saved/restored by `q`/`Q`); a matrix with
  `|det| <= 1e-12` or a non-finite product stays ambiguous.
- **Text position** — `Tm`, `Td`, `TD`, `T*`, `'`, `"` and `TL` maintain the text and line
  matrices; the device origin of every shown run is `(0, 0)` through `Tm × CTM`, and the
  effective glyph size is `font size × x-scale(Tm × CTM) × Tz/100`. `TJ` kerning numbers
  are glyph shifts, not ambiguity. A rotated or skewed but invertible `Tm` is fine.
- **Page box** — the inherited `/MediaBox` cut by any `/CropBox` (walking `/Parent`); a page
  without a valid box is ambiguous. A run whose origin is outside the page box or the clip is
  **hidden** (`PDF_HIDDEN_TEXT`), as is rendering mode 7 and an effective glyph size below
  `1e-6` pt. A glyph size below `1` pt cannot be proven to show at 300 DPI and is ambiguous.
- **Clip** — the current path is tracked; `W`/`W*` closed by a painting operator intersects
  the clip with the path only when it is a single `re` under an axis-aligned CTM. Anything else
  (curves, lines, two rectangles, an empty path, a rotated CTM) keeps `clip_ambiguous`. A clip
  narrower than the glyph size is ambiguous. A path left open across a non-path operator is
  ambiguous. A Form XObject `/BBox` is the same rectangular clip (its `/Matrix` still fails
  closed when it is not the identity).
- **Paint versus text** — every visible run is bounded by its real ink span along the baseline
  (1 em up and 0.35 em down): the current font's resolved per-glyph widths (`/Widths` for a
  simple font, `/W`/`/DW` for an Identity-H composite font, PDF spec 9.2.4/9.7.4.3) plus `Tc`
  and, once the font's byte width is known, `Tw` on code 32; a font this scanner cannot resolve
  (no `/Widths`, an unsupported `Type0` encoding, a malformed width array, a skewed `Type3`
  `/FontMatrix`) falls back to one em per glyph, the prior estimate, never a narrower guess (v11,
  `docs/runs/pdf-scanner-ambiguity-2026-09-27.md`). Interior `TJ` kerning moves ink *inside* that
  span, not past it (v10). Every fill is bounded by its path's device bbox, every stroke by that
  bbox grown by the device line width (`w`, and the miter limit `M` when the path has joins),
  images by the CTM-mapped unit square, shadings by the clip. Paint that overlaps a run by at
  least a quarter of the glyph size in both directions is ambiguous whether it comes before
  (unknown background) or after (possible cover); a background-coloured fill only counts after.
  Thinner overlaps — rules, borders, underlines — are slivers; more than eight on one run fail
  closed. Paint with an unknown bound (degenerate CTM, unknown clip for a shading) fails closed
  against every run, as in v8. Glyph ink never hides glyph ink, so a page of many runs is no
  longer ambiguous. Runs and paints are related across Form XObject boundaries in both
  directions.
- Unchanged fail-closed cases: unknown operators, malformed operands, optional content,
  `gs`, unknown colour spaces, `Tw` without a resolved simple font, `Ts` other than zero,
  `Tz` other than 100, form depth/cycles/missing resources, a rotated or mirrored run's extent
  (bounded by the clip or page, so any overlapping paint is ambiguous). `Tc` (v11) is no longer
  fail-closed — it is folded into the real advance, since it can only tighten or loosen
  inter-glyph spacing, never move already-drawn ink past its own glyph box.

Regression fixtures: `tests/service/data/known-nitra-libreoffice.qpdf.json` and
`tests/service/data/known-nitra-acceptance-2026-09-26.qpdf.json` are minimal, hand-authored
**synthetic scanner object graphs**. The historical filenames remain for compatibility;
these files do not redistribute the original acceptance pages or Nitra forms. Their only
text is `Synthetic scanner fixture`. They contain no original document text, metadata,
tagged document structure, embedded font programs or ToUnicode maps.

The first graph exercises a full-page `re W* n` clip, high positioned text, interior `TJ`
kerning and surrounding table paint. The second places text with synthetic 300-unit glyph
widths next to a column divider. Both must classify `visible_only`; independent controls
expect hidden text when the page or clip excludes its origins, and ambiguity when missing
font widths make the fallback text span cross the divider. A fixture privacy check permits
only the neutral text, basic page/font objects and the required drawing operators.

Rebuild the image and re-pin its digest after scanner runtime changes; these test fixtures
are not included in the sandbox image.

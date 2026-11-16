# API and Word layout verification — 2026-11-16

The conversion service now serves local Swagger UI at `/docs` and the shared OpenAPI contract at `/openapi.json`. It supports documented Bearer authorization, multipart `page` file uploads, raw PDF/PNG/JPEG bodies, required request settings, asynchronous polling and authenticated binary Word download. `--web-ui none` serves the API and documentation without loading application pages. EN/SK README files link `docs/API.md`; `examples/headless_client.py` reads JSON settings and an environment token.

Validation:

- Full default backend suite: 1,961 passed, 26 slow cases deselected. One existing Requests dependency warning. The host had older Pillow/certifi, so the run used task-local Pillow 13.0.0 and the image's certifi 2026.10.23 without changing host packages.
- Final client suite: 21 passed, including three additional slow-body/header deadline regressions added after the full-suite collection.
- Frontend: 646 tests passed; generated types, type check and production build passed.
- Chromium regression exercises the actual isolated API: Authorize, PDF picker, generated request key, status polling and binary Word download. It verifies no external requests, no mixed browser cookies, no saved authorization and no CSP violations.
- Deployed local FP8 service repeated the Swagger flow and the configurable headless client against the real GPU. Both two-page synthetic PDFs produced Word files with both expected page texts; Swagger had zero browser/CSP errors.
- Independent review found one total-deadline issue in the client. Three slow-response regressions reproduced it; a POSIX real-time timer now interrupts blocked network I/O, preserves prior output, removes temporary files and restores signal state.

The Word assembly repair only collapses implicit line height on childless section-owned paragraphs explicitly predicted invisible at confidence >= 95%. Text, nodes, explicit paragraph spacing, visible/uncertain paragraphs and section alignment remain intact. A local replay retained all 184 paragraphs and all text, changing only seven spacer line boxes: LibreOffice pagination improved from seven to six pages and removed the reported first-page spill. A synthetic centered case improved from two pages to one without changing alignment or content. A separate small overflow near the original document's end remains; WordCraft was not rerendered. Real input documents and credentials are excluded from the repository.

Local API and worker were updated from their existing locked images using source-only overlay layers. Accounts, model weights, existing job records and historical artifact bytes remain unchanged. Corrected historical output was written as a separate local file.

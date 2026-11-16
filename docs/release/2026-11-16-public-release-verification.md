# Public release verification — 16 November 2026

Application: Unflatten W 1.0 (`v1.0.0`). Model: Unflatten W Dense 1.0.

## Release preparation

- The complete default backend suite passed: **1,974 passed, 26 slow tests deselected**. It used task-local copies of the pinned Pillow 13.0.0 and certifi 2026.10.23, without changing the host installation. One existing Requests dependency warning remains in that test environment.
- Launcher/catalog/recovery tests passed, including fresh anonymous setup, a real interactive terminal with no credential input, start/restart, saved optional credentials and worker-only secret isolation.
- Scanner, font, sandbox, readiness and page-geometry tests passed. Two real-document-derived fixture graphs were replaced wholesale by synthetic text/geometry; a privacy guard and negative geometry controls prevent empty-content regressions.
- The unchanged frontend previously passed 646 tests, type checking and the production build. The API/Swagger release was also exercised against the real GPU; see [the preceding verification](2026-11-16-api-and-layout-verification.md).
- Full source history was scanned with Gitleaks 8.30.1: no operational credentials found. The public repository is prepared from the audited current tree, with the previous private development history preserved separately so old document blobs are not exposed.
- Every published source file is bound by `publication-manifest.json`. Model weights, credentials, databases, real input documents and local runtime data are excluded.

## Model integrity

The application retains weight-bundle revision `9ebf7df1d6b01ae5d3a396f379fe454378a38ae7` and manifest SHA-256 `0ee3a3c68e1290aeb7eaf72b9d28371c249f4d75f82851e9745ac221b0920c03`.

Updated model documentation is committed at `7ddc0af6dddb05cf08ab7cd63a6b2a75c81a2e07`. Only `README.md`, `INFERENCE.md`, `release-recipe.json` and the corresponding manifest changed. A complete candidate bundle passed `verify_bundle()`; all weight blobs, identities and licence bytes were verified unchanged. The updated manifest SHA-256 is `58091ec7259536da410a858a51477de97243abeee1ea1182c4206066d2907efc`.

## Public access

Anonymous access was verified on 16 November 2026:

- GitHub repository metadata reports public visibility, and the default-branch README links the model.
- Hugging Face reports `private=false` and `gated=false`; its model card links the application and API guide.
- Model README, updated manifest and the application's pinned configuration were retrieved without a token.
- Weight-download metadata for the 4,967,229,296-byte first backbone shard is accessible without a token.
- The initial public repository contained exactly one clean commit whose complete Git tree matched the tested release source. Previous private commits and document blobs were not imported. This verification update adds documentation only.

Public endpoints:

- [Application source](https://github.com/EvoliqCZ/UnflattenW_Inference_tool-)
- [Model and weights](https://huggingface.co/Evoliq/Unflatten-W-Dense-1.0)
- [Application v1.0.0](https://github.com/EvoliqCZ/UnflattenW_Inference_tool-/releases/tag/v1.0.0)
- [API and Swagger instructions](../API.md)

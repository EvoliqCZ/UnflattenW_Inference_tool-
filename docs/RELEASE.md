# Unflatten W 1.0 release facts

**Prepared release:** application `v1.0.0`, 16 November 2026.
**Model:** Unflatten W Dense 1.0.
**Publisher and licensor:** EVOLIQ s.r.o.

> Prepublication note: this document prepares the public release. Anonymous
> access to the model and application has not yet been confirmed. The date above
> is the prepared release date, not evidence that publication has completed.

## Model, application and documentation

- [Model card and weights on Hugging Face](https://huggingface.co/Evoliq/Unflatten-W-Dense-1.0)
- [Inference application on GitHub](https://github.com/EvoliqCZ/UnflattenW_Inference_tool-)
- [Application release listing](https://github.com/EvoliqCZ/UnflattenW_Inference_tool-/releases)
- [English setup guide](../README.md) and [Slovak setup guide](../README.sk.md)
- [API guide](API.md) and [configurable headless client](../examples/headless_client.py)
- [Application verification](release/1.0-verification.md) and
  [API/layout verification](release/2026-11-16-api-and-layout-verification.md)
- [Commercial licensing enquiries](https://www.evoliq.cz/sk)

The model's updated documentation is recorded at
[revision 7ddc0af6](https://huggingface.co/Evoliq/Unflatten-W-Dense-1.0/tree/7ddc0af6dddb05cf08ab7cd63a6b2a75c81a2e07).
That revision changes documentation and its integrity metadata, not the weights.
The application retains its tested weight-bundle pin
`9ebf7df1d6b01ae5d3a396f379fe454378a38ae7`; see the
[distributed catalogue](../deploy/distribution/models.json).

## What is included

Unflatten W reconstructs document content and structure from page images and
assembles editable Word documents. PDF input is rasterized and processed page by
page by the application. The model itself handles one raster page per call.

```text
PDF or image → page raster → Dense model → structured document → DOCX renderer → Word
```

The model combines a frozen **Qwen3-VL-4B-Instruct** visual/language backbone with
trained Unflatten document-reconstruction modules and typed property heads.
It uses `pooler_output` plus DeepStack layers 5, 11 and 17, with constrained
structural/text decoding. The deterministic renderer writes OOXML and `.docx`;
structural validation does not guarantee correct text or faithful layout.

The selected model is **Dense RL48**, continuing from the S4 parent at step
170,000. It is not a released MoE model. The standard application uses native
CUDA inference, BF16, the bundled loop guard and a BF16 dynamic KV cache. No
MTP, DSpark or other speculative-decoding speedup is claimed for this release.
No exact trainable or active parameter count is claimed beyond identifying the
Qwen3-VL-4B backbone.

The Hugging Face package includes custom weights in `unflatten_heads.safetensors`,
the frozen backbone/tokenizer/image processor, custom loading code, runtime source,
decoding files, integrity metadata and export smoke evidence. It excludes source
training/validation documents and training optimizer state. The inference
repository supplies the application and deployment source; weights are downloaded
separately. Generic chat applications and a standard text-generation pipeline do
not implement this custom reconstruction workflow.

See the [model card](https://huggingface.co/Evoliq/Unflatten-W-Dense-1.0)
and [developer loading notes](https://huggingface.co/Evoliq/Unflatten-W-Dense-1.0/blob/main/INFERENCE.md).

## Inputs, outputs and application features

| Area | Release behavior |
|---|---|
| Inputs | PDF, PNG and JPEG; one PDF or ordered image pages per job. |
| Main output | Editable Word `.docx`, with predicted text, structure and formatting. Embedded images remain image content. |
| Other artifacts | Prediction IR JSON, available PNG previews and ZIP downloads. |
| Web interface | English/Slovak setup and login, uploads, job status, per-page progress, cancellation/purge, downloads, previews and model management. |
| Headless use | Authenticated API and configurable command-line client without a browser login. `--web-ui none` omits application pages while keeping API documentation. |
| Documentation | Local Swagger UI at `/docs` and the shared schema at `/openapi.json`. No hosted public API or demo is advertised. |

The interface language switch does not translate input documents. English/Slovak
UI support is not a claim of model accuracy across languages. No per-language
quality benchmark is established by this release.

The [default runtime policy](../src/unflatten/service/policies/runtime_policy.json)
allows up to **100 pages** and **50 MiB of uploaded data**, rasterizes PDFs at
**300 DPI**, and limits raster dimensions to 6000 × 6000 pixels and 36 million
pixels. Word artifacts are limited to 64 MiB. These are admission ceilings,
not a throughput or 100-page quality guarantee. Word and prediction artifacts
have a default 24-hour retention period; metadata has a 30-day retention period.
Read the installed service's actual limits through `GET /v1/readiness`.

Model predictions can contain text errors, omissions, reading-order problems,
incorrect typography, tables or image placement, and changed pagination.
Review the generated document before relying on it. Neither a valid document
structure nor successful Word-file creation proves visual fidelity.

## Evaluation and its limits

The release selection used a diagnostic comparison on **the same fixed 100
validation pages**, not 100 documents and not an independent final test set.
S4 and RL48 used the same decoding setup. The selection cohort was reused across
candidate checkpoints; these results do not establish statistical significance
or general production quality.

| Diagnostic measure | S4 parent, step 170,000 | Dense 1.0 / RL48 |
|---|---:|---:|
| Pages passing the configured text-tolerance criterion | 89 / 100 | 90 / 100 |
| Exact text matches | 9 / 100 | 9 / 100 |
| Diagnostic Word files produced | 98 / 100 | 99 / 100 |
| Raw aggregate character error rate | 13.8698% | 13.8949% |
| Word error rate | 7.9568% | 8.1013% |

The tolerance criterion is **normalized character error rate ≤ 2%**. Its
[normalization](../src/unflatten/eval/s2_gate/score.py) collapses runs of three or
more leader characters, removes remaining underscores and removes whitespace.
The 90/100 result is therefore a page-level tolerance pass rate, not “90%
character accuracy” or exact matching. Raw character/word error rates are
separate measures.

RL48 produced one additional Word file and one additional tolerance pass,
while aggregate text error rates were slightly higher than the parent.
Rendered visual fidelity was not scored. Typed-property alignment covered only
7/100 pages for these two arms and must not be presented as full-cohort property
accuracy. The formal RL stage gate did not pass because the matched SFT
continuation comparison arm was absent; `product_quality_claim` remains `false`.

An additional offline export check on NVIDIA DGX Spark used **three validation
pages**. All produced strictly valid structures and Word files, and accepted
tokens matched the previously evaluated RL48 tokens exactly. This is export
compatibility evidence, not another model-quality benchmark. See the
[model-card evaluation](https://huggingface.co/Evoliq/Unflatten-W-Dense-1.0#evaluation)
and [export evidence](https://huggingface.co/Evoliq/Unflatten-W-Dense-1.0/blob/main/export-smoke-test.json).

Application tests, browser checks and synthetic GPU conversions are engineering
acceptance evidence, separate from these model-quality measurements. Their
counts and timings must not be combined into an accuracy or throughput claim.

## Platform and getting started

The documented application deployment requires Linux, Docker Engine with Compose,
NVIDIA Container Toolkit and a compatible NVIDIA CUDA driver. The locked runtime
uses CUDA 13. **At least 16 GB GPU memory is recommended for the standard BF16
model**, with loading subject to actual available memory. Budget about 11 GB for
model files, plus container images and documents.

BF16 export and historical application acceptance were exercised on NVIDIA DGX
Spark GB10/Linux aarch64. Later RTX 4070 SUPER 12 GB acceptance used a **separate
local FP8 test bundle**, which is not distributed or automatically selected by
this release. It does not establish that the standard BF16 model fits a 12 GB
GPU. Other platforms need their own validation; no CPU, Windows or macOS support
claim is made here.

Use the current application lockfiles and setup guide. The model's historical
export environment record reused installed dependencies and is not a fresh
installation certification for the current application.

```bash
git clone https://github.com/EvoliqCZ/UnflattenW_Inference_tool-.git
cd UnflattenW_Inference_tool-
./Start.sh
```

Follow the launcher, wait for model preparation, create an application password,
upload a document and review the resulting Word file. Initial building and model
download require internet access. With weights downloaded, inference processes
documents locally. PDFs are parsed in a separate restricted container. This is
not a claim that the entire application never makes network requests.

For headless conversion, follow the [API guide](API.md). The workflow is:

1. Authorize using the service's API Bearer token.
2. Submit `POST /v1/conversions` with an idempotency key, profile `word-core-v0`
   and output `diagnostic_docx`.
3. Poll `GET /v1/conversions/{job_id}` until terminal.
4. After `succeeded` and `diagnostic_outcome.docx_produced=true`, download
   `GET /v1/conversions/{job_id}/artifacts/diagnostic_docx`.

The API token, browser password and Hugging Face download token are distinct
credentials. Swagger keeps authorization in memory, omits browser cookies and
serves its assets locally. The example JSON-configured client requires Python
3.12+ on Linux/POSIX and uses only the standard library. It enforces a total
deadline and replaces output atomically only after a complete download. Client
timeout does not cancel the server job. Optional request settings select an
existing model variant and document label; they do not expose arbitrary prompts,
temperature, DPI, quantization or generation controls.

## Licensing

EVOLIQ's original model contributions, code and documentation use the
[EVOLIQ Personal Use and Commercial License 1.0](../LICENSE): free private,
noncommercial personal use by an individual. Employer/client work, internal
organizational use, professional work, product integration and hosted services
require a separate paid written commercial agreement with EVOLIQ s.r.o. Pricing
and scope are agreed separately through the [company contact form](https://www.evoliq.cz/sk).

The unmodified Qwen backbone retains **Apache License 2.0**. Independent
third-party rights are unchanged; see [NOTICE](../NOTICE) and
[LICENSE-APACHE-2.0](../LICENSE-APACHE-2.0). This is a source-available release
under the supplied personal/commercial terms, not an unrestricted commercial-use
or whole-package Apache-2.0 grant.

## Release notes — v1.0.0

- Local PDF/PNG/JPEG conversion to Word, an English/Slovak web interface, job
  management, previews, and verified model download/load workflows.
- A documented Bearer-authenticated API with raw and multipart upload,
  asynchronous status and binary artifact download, local Swagger and OpenAPI.
- Headless server mode and a configurable Python client with bounded waiting,
  explicit failures and safe output replacement.
- Multi-page image/resource handling and a conservative Word-layout repair for
  high-confidence invisible empty paragraphs. The repair preserves content and
  section alignment; it is not a general pagination-fidelity guarantee.
- Existing Dense RL48 weights and licensing retained. The standard model is BF16;
  MoE and speculative accelerators are not released features.

Detailed test scope and remaining limitations are in the
[application verification](release/1.0-verification.md) and
[API/layout verification](release/2026-11-16-api-and-layout-verification.md).

## Slovenský text pre web

**Unflatten W Dense 1.0 — z PDF a obrázkov do editovateľného Wordu**

Unflatten W od EVOLIQ s.r.o. rekonštruuje text a štruktúru dokumentov do formátu
Word. Lokálna aplikácia pre Linux s NVIDIA GPU ponúka rozhranie v slovenčine a
angličtine, správu konverzií, náhľady a API so Swagger dokumentáciou.

Výsledný dokument treba skontrolovať: text, tabuľky, obrázky aj zalomenie strán
môžu obsahovať chyby. Osobné nekomerčné použitie je bez licenčného poplatku;
firemné, profesionálne a komerčné použitie vyžaduje platenú dohodu s EVOLIQ s.r.o.

[Model na Hugging Face](https://huggingface.co/Evoliq/Unflatten-W-Dense-1.0) ·
[Inferenčná aplikácia](https://github.com/EvoliqCZ/UnflattenW_Inference_tool-) ·
[Komerčná licencia](https://www.evoliq.cz/sk)

Pred zverejnením odkazu na stiahnutie overte verejnú dostupnosť oboch repozitárov.
Dátum prípravy tejto stránky sám osebe nepotvrdzuje dokončenie publikovania.

## Public demonstration assets

Use supplied product/company artwork or purpose-made synthetic documents for
public demonstrations. Label UI-only fixture demonstrations separately from
actual GPU/model outputs. Do not publish customer documents, validation inputs,
saved predictions or screenshots containing credentials. Historical architecture
and timing material is not release evidence; use the facts and limitations above.

# Unflatten W API

The service converts local PDF, PNG and JPEG files asynchronously. Machine
clients need no browser registration, login or session cookie. The API and worker
must be running, with the configured model and PDF sandbox ready.

Default local service: `http://127.0.0.1:8421`. Interactive documentation:
[/docs](http://127.0.0.1:8421/docs). OpenAPI:
[/openapi.json](http://127.0.0.1:8421/openapi.json). Change the port in these URLs
when your deployment uses another port. Swagger assets are served locally; API
tokens are not persisted by Swagger.

## Authentication

Requests use `Authorization: Bearer <API token>`. This token identifies a
principal and its permissions; jobs and downloads belong to that principal.
It is separate from the application password and the Hugging Face token used to
download weights. Web login is unnecessary for headless clients.

The local initializer creates a private `principals.json` with an API token and
these permissions: `submit_diagnostic`, `read_status`, `download_prediction_ir`,
`cancel_own_job`, `download_docx`. It preserves an existing file and does not print
the token in distribution startup logs. There is no public token-issuing endpoint.

A local administrator can explicitly display the existing `local` principal's
token with this command, run from the repository directory. Use the saved
configuration of the intended deployment:

```bash
# Default distribution; change this path if you selected another data directory.
APP_ENV="$HOME/.local/share/unflatten-w-app-1.0.launcher/app.env"
docker compose --env-file "$APP_ENV" -f deploy/distribution/compose.yaml \
  exec -T api python3 - <<'PY'
import json
import os

with open(os.environ["UNFLATTEN_PRINCIPAL_FILE"]) as source:
    principals = json.load(source)["principals"]
print(next(item["token"] for item in principals if item["principal_id"] == "local"))
PY
```

The command prints a real credential only when you run it. Keep that terminal
private. For another Compose project, use its own environment file, Compose file
and project name. The credential is in that service's `UNFLATTEN_PRINCIPAL_FILE`,
not the adjacent launcher directory holding its Hugging Face secret.

Supply it to the examples without placing it in shell history or request JSON:

```bash
read -r -s -p 'API token: ' UNFLATTEN_API_TOKEN
printf '\n'
export UNFLATTEN_API_TOKEN
```

For an unattended caller, inject the environment variable from your private
credential store. Do not include credentials in URLs, source control or logs.
Native operators provision the same principal-file format, owned by the API
service user with mode `0400`; see [native deployment](../deploy/service/README.md).
After changing that file, restart the API to load the new credentials.

## Swagger walkthrough

1. Open `/docs`, choose **Authorize**, and paste only the API token, without the
   `Bearer ` prefix. Swagger adds the header. This uses API authentication
   independently of any application login; the documentation client omits browser
   cookies.
2. Expand **POST /v1/conversions** and choose **Try it out**. Keep profile
   `word-core-v0` and output `diagnostic_docx`. With the default
   `multipart/form-data` request, choose a PDF or page image in the **page** file
   picker. Leave optional model/label fields empty unless needed.
3. Keep `Idempotency-Key` as `auto`: Swagger generates a new UUID on each
   **Execute**. For a deliberate retry of the same upload, enter your own fixed
   key. The literal `auto` is a Swagger convenience, not a headless client policy.
4. Submit and copy `job_id` from the `202` response. Execute
   **GET /v1/conversions/{job_id}** until `state` is terminal. Word is available
   when `state` is `succeeded` and `diagnostic_outcome.docx_produced` is `true`.
5. Execute **GET /v1/conversions/{job_id}/artifacts/diagnostic_docx** with the
   same authorization to download the binary `.docx`. The ZIP endpoint downloads
   all available artifacts. Pasting an artifact URL into an unauthenticated tab
   does not supply the Bearer token.

Swagger does not poll automatically. Reloading the documentation clears its API
authorization.

## Supported request settings

The upload is the request body, not a JSON object. JSON configuration belongs to
the example client below; it is not the wire format of `POST /v1/conversions`.

| Setting | Meaning |
| --- | --- |
| `Authorization` | Required `Bearer <API token>` on protected requests. |
| `Idempotency-Key` | Required upload key, 1–256 printable ASCII characters without spaces. Use a new UUID for a new conversion. Reuse a key for an identical retry; conflicting reuse returns `409`. |
| `X-Unflatten-Profile` | Required: `word-core-v0`. |
| `X-Unflatten-Output` | Required: `diagnostic_docx` for Word, or `prediction_ir` for structured prediction only. |
| `Content-Type` | Raw `application/pdf`, `image/png`, `image/jpeg`, or `multipart/form-data` with a `page` file field. Let your multipart client generate its boundary. |
| `X-Unflatten-Model` | Optional exact `variant_id` from `GET /v1/models`. Omit to use the effective default. The variant must be available and supported by the worker/GPU. |
| `X-Unflatten-Label` | Optional percent-encoded UTF-8 document label, 1–120 characters after decoding/normalization. Filesystem-hostile and control characters are rejected. The Python client encodes it for you. |

The existing Word identifier is `diagnostic_docx`. The value `docx` returns
`503 CAPABILITY_NOT_READY`; the API does not promise product-quality or exact
layout reconstruction. Review the generated text and layout. Temperature,
prompts, DPI, output format, arbitrary decoding options and precision changes are
not configurable per conversion. A model header selects an existing configured
variant; it does not quantize weights or create a new model.

Inspect current limits and readiness with authenticated `GET /v1/readiness`
(`limits.max_pages`, `limits.max_upload_bytes`, `limits.accepted_media_types`).
`GET /v1/models` reports availability, memory fit, download state and the resident
model. Existing model-management endpoints can download/load configured variants
and set a default; use their documented schemas in Swagger. The public
`GET /health/ready` reports operational readiness without credentials.

## Headless client with JSON configuration

The [standalone client](../examples/headless_client.py) requires Python 3.12 or
later on Linux/POSIX and uses only the standard library. A real-time signal
timer enforces the total deadline even during slow uploads, headers or response
bodies; platforms without that timer fail explicitly. Copy the
[configuration](../examples/headless-request.example.json)
to your working directory and edit it:

```json
{
  "base_url": "http://127.0.0.1:8421",
  "input": "document.pdf",
  "output": "result.docx",
  "label": "Example document",
  "poll_interval": 2,
  "timeout": 1800
}
```

`base_url`, `input` and `output` are required. Paths are relative to the JSON
file, or absolute. Optional `model` is a variant ID from `/v1/models`; omit it
to keep the service's default. `label` is plain Unicode here. `poll_interval`
defaults to 2 seconds and `timeout` to 1,800 seconds; both must be positive.
The timeout covers upload, polling and download. Only local file inputs are
supported; the client does not fetch source documents from URLs.

```bash
cp examples/headless-request.example.json request.json
# Edit request.json and set UNFLATTEN_API_TOKEN as above.
python3 examples/headless_client.py request.json
unset UNFLATTEN_API_TOKEN
```

Each invocation creates a fresh UUID, streams the raw input, polls the job and
downloads Word with the same Bearer token. It exits nonzero for HTTP/network
errors, a timeout, `failed`, `cancelled`, `expired`, or success without Word.
A complete download replaces the configured output atomically; incomplete
downloads leave an existing output intact. A client timeout does not cancel the
server job. It prints the job ID so you can inspect or cancel that job later.
It never follows redirects or sends credentials to URLs supplied in a response.

## Raw curl requests

With `UNFLATTEN_API_TOKEN` set privately, submit a PDF:

```bash
BASE_URL=http://127.0.0.1:8421
REQUEST_KEY="$(python3 -c 'import uuid; print(uuid.uuid4())')"
curl --fail-with-body -sS -X POST "$BASE_URL/v1/conversions" \
  -H "Authorization: Bearer $UNFLATTEN_API_TOKEN" \
  -H "Idempotency-Key: $REQUEST_KEY" \
  -H 'X-Unflatten-Profile: word-core-v0' \
  -H 'X-Unflatten-Output: diagnostic_docx' \
  -H 'Content-Type: application/pdf' \
  --data-binary @document.pdf
```

For PNG/JPEG use its matching content type. For multipart, replace the content
type and `--data-binary` options with `-F 'page=@document.pdf;type=application/pdf'`.
The API returns `202`, `job_id`, `status_url` and a `Location` header. Set
`JOB_ID` to the returned identifier, then inspect status:

```bash
JOB_ID=job_returned_identifier
curl --fail-with-body -sS "$BASE_URL/v1/conversions/$JOB_ID" \
  -H "Authorization: Bearer $UNFLATTEN_API_TOKEN"
```

States are `queued`, `running`, `succeeded`, `failed`, `cancelled`, `expired`.
Status includes `stage`, page progress, `diagnostic_outcome`, `artifacts` and
`error`. Poll with a delay and deadline as the Python example does. After
`succeeded` plus `diagnostic_outcome.docx_produced=true`:

```bash
curl --fail -sS "$BASE_URL/v1/conversions/$JOB_ID/artifacts/diagnostic_docx" \
  -H "Authorization: Bearer $UNFLATTEN_API_TOKEN" --output result.docx
```

This simple curl download writes directly to the output; the Python client
provides atomic replacement. Additional protected routes:

| Method and path | Purpose |
| --- | --- |
| `GET /v1/conversions` | List this principal's jobs. |
| `DELETE /v1/conversions/{job_id}` | Request cancellation. |
| `DELETE /v1/conversions/{job_id}?mode=purge` | Delete a finished job and its artifacts. |
| `GET /v1/conversions/{job_id}/artifacts.zip` | Download available artifacts together. |
| `GET /v1/conversions/{job_id}/artifacts/prediction_ir` | Download structured prediction; page-scoped URLs are listed in status when applicable. |

`401` means missing/invalid credentials, `403` insufficient permissions,
`404` absent or foreign resources, `410` expired artifacts, `413` an oversized
upload, `422` an unavailable/unsuitable requested model, and `503` an unavailable
capability or full queue. Inspect the problem response's `code`; never interpret
an HTTP failure body as Word bytes.

## Server without the application UI

API clients work with every UI mode. To omit the application pages entirely,
native operators add `--web-ui none` to their configured
`python -m unflatten.service.api_cli` command, or set `UNFLATTEN_WEB_UI=none`.
Keep the other principal, database, artifact, model and worker settings unchanged.
This mode serves the API, `/docs`, its local assets and `/openapi.json` without
loading a built SPA.

For an already provisioned lower-level [Docker deployment](../deploy/docker/README.md),
the following recreates only its API with that mode:

```bash
cd deploy/docker
UNFLATTEN_WEB_UI=none docker compose -f compose.yaml -f compose.sandbox-socket.yaml up -d api
```

Use that deployment's existing environment, model and sandbox configuration.
The standard `Start.sh`/`Spustit.sh` distribution Compose file explicitly selects
`spa`; setting this variable in its shell does not override that file.

The default service remains bound to loopback. For another machine, keep the
service private behind your configured SSH tunnel or authenticated TLS proxy.
Preserve the Host allowlist and existing deployment authentication rules; API
documentation does not widen network binding or enable CORS.

## Slovenský postup

1. Spustite API a worker s pripraveným modelom. Pre priame API volania nepotrebujete
   webovú registráciu ani heslo. Použite interný API token z `principals.json`,
   oddelený od HF tokenu; správca ho môže zobraziť vyššie uvedeným príkazom.
2. V `/docs` kliknite **Authorize**, vložte samotný token a otvorte
   **POST /v1/conversions**. Vyberte súbor v poli **page**, ponechajte
   `word-core-v0`, `diagnostic_docx` a kľúč `auto`. Každé **Execute** vytvorí
   nový identifikátor požiadavky.
3. Z odpovede skopírujte `job_id` a sledujte **GET /v1/conversions/{job_id}**.
   Po `succeeded` a `docx_produced=true` stiahnite Word cez príslušnú trasu
   `artifacts/diagnostic_docx` s rovnakou autorizáciou.
4. Pre automatické spracovanie upravte `request.json`, nastavte
   `UNFLATTEN_API_TOKEN` a spustite `python3 examples/headless_client.py request.json`.
   Nastaviť možno vstup, výstup, adresu služby, existujúci model, názov a časovanie.
   Klient sám odošle dokument, počká na výsledok a uloží Word bez rozhrania.
5. Pre server úplne bez aplikačných stránok použite v podporovanom priamom
   nasadení `--web-ui none` podľa postupu vyššie. Swagger a API zostanú dostupné;
   predvolené lokálne sieťové obmedzenia zostávajú zachované.

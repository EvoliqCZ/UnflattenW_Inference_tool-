# Diagnostická služba cez `docker compose` (DEC-044 vrátane Dodatku D; webová aplikácia a
# distribuovateľný balík — DEC-052, PROPOSED do formálneho T50 security review)

Prenosné spustenie privátnej diagnostickej PDF→Word služby na ľubovoľnom stroji s jednou
NVIDIA GPU + NVIDIA Container Toolkit, bez manuálnej inštalácie Pythonu/uv na hostiteľa.
Toto je alternatíva k `deploy/service/service.sh` (ten ostáva nezmenený a funguje priamo na
hostiteľovi); oba spôsoby spúšťajú tie isté `unflatten.service.api_cli` / `worker_cli`.

For the released application, use the [English installation guide](../../README.md)
or [Slovak guide](../../README.sk.md). Current lifecycle commands are in the
[application runbook](../../docs/RUNBOOK.md). This document covers the lower-level
Compose configuration for operators who need to customize it.

**Vlastníkovo pravidlo: obrazy si každý stavia sám, žiadny registry push.** Neexistuje
zdieľaný pre-buildnutý obraz tejto služby ani sandboxu — `docker compose build` (alebo
`deploy/local/run-local.sh up`) postaví `api`/`worker`/sandbox lokálne z tohto zdroja pri
každom nasadení. Build webovej SPA (`web-builder` Docker stage) potrebuje pri stavbe obrazu
**dosiahnuteľný npm registry** (`npm ci` sťahuje z `registry.npmjs.org`, pokiaľ nie je
nastavený firemný proxy/registry v `.npmrc`); offline/air-gapped stavba obrazu vyžaduje
vlastnú npm cache/registry mirror mimo rozsahu tohto README.

## Súbory

| Súbor | Účel |
|---|---|
| `Dockerfile` | multi-stage; `web-builder` (Node 22 LTS, SPA build) + dva runtime ciele z jedného Python zdroja: `runtime-api` (malý), `runtime-worker` (+ LibreOffice/poppler/Carlito, viď nižšie) |
| `entrypoint-api.sh` | mapuje env premenné na `api_cli` argumenty (JEDNO miesto) |
| `entrypoint-worker.sh` | mapuje env premenné na `worker_cli` argumenty (JEDNO miesto) |
| `entrypoint-init.sh` | jednorazovo vygeneruje principal file s náhodným tokenom |
| `compose.yaml` | `api` + `worker` + `init` + `sandbox-image`, projekt `unflatten-diagnostic` |
| `compose.sandbox-socket.yaml` | **voliteľný** override: dá workeru prístup k hostiteľskému docker socketu |
| `.env.example` | šablóna — skopíruj na `.env` v tomto adresári |
| `models.example.json` | šablóna pre `models.json` (viď „Model" nižšie); validovaná `unflatten.service.models_config.load_models_config`. BF16/FP8/INT8 sú `"availability": "unpublished"` (HF repozitáre ešte neexistujú, DEC-050), Unflatten W MoE je `"planned"`; po zverejnení variantu odstráň `availability` a doplň `source` s pripnutým `revision` a `bundle_manifest_sha256` |
| `models.json` | **negenerovaný, negitovaný** — píše ho `deploy/local/dev-bundle.sh`; `deploy/local/run-local.sh up` sa bez neho odmietne spustiť |

## LibreOffice/poppler: iba v `runtime-worker`, nie v `runtime-api`

`docs/RUNBOOK.md` §10: „Hostiteľ nepotrebuje poppler ani LibreOffice — obe žijú iba v
sandbox obraze / lokálnej prijímacej kontrole" — to platí pre nedôveryhodnú PDF/obrázkovú
admisiu a rasterizáciu, ktorá beží výhradne v oddelenom `deploy/pdf-sandbox` obraze,
spúšťanom per-job cez `PdfSandbox` (`src/unflatten/service/pdf_sandbox.py`), a preto ju
`runtime-api` cieľ vôbec neinštaluje.

`runtime-worker` cieľ navyše obsahuje `libreoffice-writer`, `poppler-utils` a
`fonts-crosextra-carlito` (metricky zhodná náhrada Calibri, DEC-044 Dodatok C) pre
náhľadovú cestu DOCX → PNG (worker renderuje VLASTNÝ, už zostavený DOCX pre porovnanie
vedľa originálu v UI — nie nedôveryhodný vstup, teda mimo hostile-PDF sandboxu).
LibreOffice potrebuje zapisovateľný per-proces profilový adresár
(`-env:UserInstallation=`, `src/unflatten/renderer/probe.py`, `$HOME/unflatten-probe`);
`HOME=/home/unflatten` je vytvorený a vlastnený `unflatten` používateľom (alebo je
zapisovateľný pre root pri `compose.sandbox-socket.yaml`), `/tmp` má predvolený mód 1777.
Náhrada Calibri→Carlito (`src/unflatten/renderer/fonts.py`) je síce iba zápis mena písma
do OOXML, ale skutočný render náhľadu už fyzicky potrebuje font s daným menom nainštalovaný
— odtiaľ `fonts-crosextra-carlito` v tomto cieli.

## Rýchly štart (bez GPU, bez sandboxu — čo overuje samo seba)

```bash
cd deploy/docker
cp .env.example .env
# vyplň aspoň UNFLATTEN_MODEL_REVISION a UNFLATTEN_SANDBOX_IMAGE (aj placeholder stačí
# na to, aby worker vedel zlyhať SPRÁVNYM dôvodom -- viď nižšie)
mkdir -p models   # alebo nastav UNFLATTEN_HOST_MODEL_BUNDLE_DIR na skutočný balík

docker compose config          # validácia compose súboru
docker compose build           # postaví obraz
docker compose up init         # vygeneruje principal file, vytlačí token RAZ do logu
docker compose logs init       # tu je bearer token
docker compose up -d api       # /health/live musí odpovedať 200
curl -sS http://127.0.0.1:${UNFLATTEN_PORT:-8000}/health/live

docker compose up worker       # BEZ compose.sandbox-socket.yaml toto ZLYHÁ okamžite --
                                # PdfSandbox.require_ready() beží pred načítaním modelu
                                # (worker_cli.main), takže sa to dá overiť aj bez GPU.
docker compose logs worker     # očakávaj chybu o nedostupnom sandbox runtime, NIE pokus
                                # o načítanie modelu
```

### Prvé nastavenie webového UI bez tokenu (vlastník 2026-09-29, DEC-052 Dodatok A)

Port sa publikuje na `${UNFLATTEN_BIND_ADDRESS:-127.0.0.1}` (nová premenná v `.env`).
`deploy/local/run-local.sh up` z nej odvodí `UNFLATTEN_SETUP_TOKENLESS` a exportuje ho pre
compose (prebije hodnotu z `.env`):

- **loopback** (`127.0.0.1`, predvolené) → `1`: sprievodca `/setup` pýta len nové heslo;
  `UNFLATTEN_SETUP_TOKENLESS=0` v `.env` token vynúti aj tu (SSH tunel / proxy);
- **iná adresa** (LAN IP, `0.0.0.0`) → `0` a skript vypíše odkaz
  `http://<host>:<port>/setup#token=<t>` (token z `docker compose logs init`, vo fragmente —
  do servera ani logu sa nedostane, SPA ho po prečítaní zmaže z adresného riadku);
  `UNFLATTEN_SETUP_TOKENLESS=1` s neloopback adresou skript odmietne (`rc=2`). LAN host treba
  pridať aj do `UNFLATTEN_WEB_ALLOWED_HOSTS`; API s tokenless a neloopback allowlistom
  odmietne štart.

Holý `docker compose up` (bez skriptu) má v compose predvolené `UNFLATTEN_SETUP_TOKENLESS=0`
— token ostáva vyžadovaný. `entrypoint-api.sh` mapuje `1` na `api_cli --web-setup-tokenless`.
Tokenless setup je jednorazový (po uložení hesla `409 SETUP_ALREADY_COMPLETE`); zabudnuté
heslo rieši len `run-local.sh reset-password`.

Toto je presne to zamýšľané správanie bez override súboru: worker odmietne prevziať
akúkoľvek úlohu, kým sandbox nie je pripravený, a hlási to vlastnou readiness správou
namiesto tichého zlyhania neskôr.

## Plné spustenie so sandboxom (potrebuje GPU + pripravený sandbox obraz)

```bash
# 1. Postav/pripni sandbox obraz NA HOSTITEĽOVI (nie cez compose -- RUNBOOK §10):
docker build -f ../pdf-sandbox/Containerfile -t unflatten/pdf-sandbox:local ../pdf-sandbox
docker image inspect --format '{{.Id}}' unflatten/pdf-sandbox:local
# → zapíš do .env: UNFLATTEN_SANDBOX_IMAGE=sha256:<id>

# 2. Postav balík modelu NA HOSTITEĽOVI (potrebuje torch, teda uv/ml skupinu lokálne,
#    nie tento kontajner -- viď scripts/build_model_bundle.py):
uv run python scripts/build_model_bundle.py build \
  --checkpoint runs/s4-v15b/checkpoint-step-000027915.pt \
  --run-dir runs/s4-v15b --out ./models
uv run python scripts/build_model_bundle.py verify ./models
# → zapíš vytlačenú "manifest sha256" do .env ako UNFLATTEN_MODEL_BUNDLE_SHA256

# 3. Nastav UNFLATTEN_HOST_DATA_DIR na ABSOLÚTNU cestu (pozri varovanie v
#    compose.sandbox-socket.yaml -- musí sa zhodovať vnútri kontajnera aj na hostiteľovi,
#    lebo worker spúšťa sandbox kontajnery cez hostiteľský docker socket, teda "súrodenca",
#    nie vnorený kontajner -- ten vidí LEN hostiteľské cesty).
mkdir -p /home/$USER/unflatten-diagnostic-data
echo 'UNFLATTEN_HOST_DATA_DIR=/home/'"$USER"'/unflatten-diagnostic-data' >> .env

docker compose -f compose.yaml -f compose.sandbox-socket.yaml up
```

**Bezpečnostné varovanie** (opakuje sa v `compose.sandbox-socket.yaml` a v
`docs/RUNBOOK.md` §10): pripojenie `/var/run/docker.sock` do workera je ekvivalent roota na
hostiteľovi. Použi to len na jednopoužívateľskej pracovnej stanici, ktorej dôveruješ; nikdy
na zdieľanom alebo sieťovo vystavenom stroji. Bez tohto override súboru worker beží ako
neprivilegovaný účet (`unflatten`, uid 65532) a nemá k soketu prístup vôbec.

## Model: kompletný balík (DEC-050) vs. staršia dvojdielna forma

**DEC-050** (2026-09-27): model sa odteraz ponúka ako **jeden kompletný balík** —
`UNFLATTEN_MODEL_BUNDLE` (bind-mount `./models` → `/models`, read-only) nesie chrbticu
Qwen3-VL pod `backbone/` AJ naše váhy/manifest z `build_model_bundle.py` v jednom adresári,
načítaný úplne offline z balíka. V tomto tvare **zdieľaná HF cache nie je potrebná** — nič sa
z Hugging Face nesťahuje a `UNFLATTEN_HF_HUB_OFFLINE`/`HF_HOME`/`HF_TOKEN` ostávajú nevyužité.

Starší (pred-DEC-050) tvar balíka bez `backbone/` ostáva spätne kompatibilný: chrbtica sa
naťahuje z Hugging Face do `HF_HOME` (predvolene `$UNFLATTEN_DATA_DIR/hf-cache`, teda vo
`unflatten_data` volume alebo v bind-mounte pri `compose.sandbox-socket.yaml`) — pri prvom
štarte preto treba sieť (`UNFLATTEN_HF_HUB_OFFLINE=0`, predvolené), kým sa pripnutá revízia
nestiahne; potom môžeš nastaviť `UNFLATTEN_HF_HUB_OFFLINE=1` pre plne offline beh (DEC-044
bod 6). Toto compose nastavenie nerozlišuje oba tvary v kóde — `entrypoint-worker.sh` iba
vždy pripraví `HF_HOME`, ktorý kompletný balík jednoducho nepoužije.

Výber viacerých
variantov modelu (`UNFLATTEN_MODELS_CONFIG`/`UNFLATTEN_PRECISION`/`UNFLATTEN_KV_CACHE`, DEC-044
Dodatok D bod 3-4) je práca paralelného agenta, ktorá ešte nepristála na `worker_cli`; env
premenné sa už prenášajú cez `entrypoint-worker.sh` (`export`), ale nemapujú sa na žiadny CLI
prepínač, kým ten neexistuje — jediné miesto na doplnenie mapovania je práve
`entrypoint-worker.sh` (komentár tam to opakuje).

## Overenie bez GPU (urobené v tejto zmene)

- `docker compose config` — validný (obe kombinácie base aj s `compose.sandbox-socket.yaml`)
- `docker build` cez `docker compose build` — obraz sa postaví
- `api` kontajner: `GET /health/live` → 200 po štarte
- `init` kontajner: vygeneruje principal file 0400 s jedným tokenom, druhý beh ho nechá
  nedotknutý (idempotentné). **Oprava (2026-09-27):** predchádzajúci skript zapisoval holé
  JSON pole namiesto tvaru, ktorý vyžaduje `auth.py` (`_validate_payload`/`load_principal_store`:
  objekt s `principal_file_schema_version` a `principals`), takže API kontajner sa vôbec
  nenaštartoval — netestované tvrdenie vyššie bolo nesprávne. Opravené a pokryté testom, ktorý
  spúšťa skript a načíta výsledok cez `load_principal_store`.
- `worker` kontajner BEZ `compose.sandbox-socket.yaml`: zlyhá okamžite s dôvodom
  "sandbox unready", nikdy sa nepokúsi načítať model — potvrdzuje fail-closed cestu
- `hadolint`/`shellcheck` (spustené cez `docker run`, nie systémovo)

`GET /` (web UI) je odteraz súčasťou obrazu — `api` servíruje buď `legacy` (pôvodná
jedna stránka), alebo `spa` (React aplikácia z `web/`, postavená stage-om `web-builder`
pri každom `docker compose build`), podľa `UNFLATTEN_WEB_UI` (viď „Web UI" nižšie); žiadny
reverse proxy medzi ním a `api_cli` v tomto compose.

## Web UI (`UNFLATTEN_WEB_UI`, DEC-052, PROPOSED)

`.env.example` má `UNFLATTEN_WEB_UI=legacy` (predvoľba sa nemení týmto behom — prepnutie na
`spa` je úloha T52, striktne po formálnom T50 security review PASS). Plný popis oboch UI
ciest je v `docs/RUNBOOK.md` §10 „Lokálne webové UI"; tu iba to, čo je špecifické pre build
obrazu: `web-builder` stage je `npm --prefix web ci && npm --prefix web run build` nad
digest-pinovaným `node:22-bookworm-slim` — **vyžaduje dosiahnuteľný npm registry pri
stavbe obrazu** (pozri varovanie na začiatku tohto README); `web/dist` sa nikdy necommituje
(`web/.gitignore`), takže každá stavba obrazu ho generuje odznova.

## Čo vyžaduje GPU (nespúšťané v tejto zmene, presný zoznam príkazov)

```bash
# predpoklad: sandbox obraz postavený + pripnutý (krok 1 vyššie), balík modelu (krok 2),
# UNFLATTEN_HOST_DATA_DIR nastavené (krok 3), .env kompletné
docker compose -f compose.yaml -f compose.sandbox-socket.yaml up -d

curl -sS -X POST "http://127.0.0.1:${UNFLATTEN_PORT:-8000}/v1/conversions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Idempotency-Key: $(uuidgen)" \
  -H "X-Unflatten-Profile: word-core-v0" \
  -H "X-Unflatten-Output: diagnostic_docx" \
  -H "Content-Type: application/pdf" \
  --data-binary @page.pdf

# potvrdzuje: worker prejde sandbox.require_ready() aj model load, GPU pamäť
# (torch.cuda.max_memory_allocated()) a čas reťazca zodpovedajú docs/runs/service-acceptance-2026-09-26.md
docker compose -f compose.yaml -f compose.sandbox-socket.yaml logs worker
```

## Predpoklady o mene env premenných na zosúladenie

Táto zmena predpokladá dnešné, skutočne existujúce mená (`UNFLATTEN_MODEL_ID`,
`UNFLATTEN_MODEL_REVISION`, `--model-id`/`--revision` na `worker_cli`). Keď paralelná práca na
kvantizovaných variantoch pristane s iným menom alebo tvarom (`--models-config` namiesto
`--model-id`/`--revision`), zmeň iba `entrypoint-worker.sh` — `compose.yaml`,
`compose.sandbox-socket.yaml` a `.env.example` už menované premenné (`UNFLATTEN_MODELS_CONFIG`,
`UNFLATTEN_PRECISION`, `UNFLATTEN_KV_CACHE`) prenášajú do kontajnera nezmenené.

# Lokálne spustenie diagnostickej služby (jedna pracovná stanica, ľubovoľná NVIDIA GPU)

Tri skripty (T47, design §8.5, DEC-052 nahrádza DEC-044 Dodatok D §6):

| Skript | Čo robí |
|---|---|
| `deploy/local/run-local.sh` | **`docker compose`** obal (design §8.5) — `up\|down\|status\|logs\|reset-password\|verify`, `--dry-run` |
| `deploy/local/dev-native.sh` | pôvodný obsah tohto adresára (predtým `run-local.sh`) — spúšťa `deploy/service/service.sh` PRIAMO na hostiteľovi, bez Dockeru |
| `deploy/local/dev-bundle.sh` | postaví lokálne varianty balíka modelu a zapíše `deploy/docker/models.json` pre `run-local.sh up` |

Predtým bol `run-local.sh start` jediný príkaz. **Zmizol** — vlastník ho používal denne, takže
namiesto ticha vypíše nápovedu (`run-local.sh start` → rc 2, odkaz na `dev-native.sh start`
alebo `run-local.sh up`).

## `run-local.sh` — docker compose

```bash
deploy/local/dev-bundle.sh                # postaví lokálne varianty modelu, zapíše deploy/docker/models.json
deploy/local/run-local.sh up              # docker info, .env, rootful-socket súhlas, models.json, build, up -d
deploy/local/run-local.sh status          # docker compose ps + /health/ready
deploy/local/run-local.sh logs            # docker compose logs -f
deploy/local/run-local.sh reset-password  # exec do `api`, web_auth_cli reset-password (principal_id "local")
deploy/local/run-local.sh down            # docker compose down -- NIKDY -v (dáta ostávajú)
deploy/local/run-local.sh verify          # docker compose config -q + sandbox image scanner identity
deploy/local/run-local.sh up --dry-run    # vypíše poradie krokov a príkazy, nič nespustí
```

`up`, v poradí (každý krok beží skutočne, aj pri `--dry-run` — iba `docker compose build`/
`up -d` sa pri `--dry-run` iba vypíšu):

1. `docker info` — daemon dosiahnuteľný, inak presná správa (RUNBOOK §10);
2. `deploy/docker/.env` chýba → skopíruje sa z `.env.example`, doplní
   `UNFLATTEN_HOST_DATA_DIR=$HOME/unflatten-diagnostic-data` a
   `UNFLATTEN_GIT_SHA=$(git rev-parse --short HEAD)`;
3. **`UNFLATTEN_ACCEPT_ROOTFUL_SOCKET=1`** je jediný súhlasný príznak, ktorý treba nastaviť
   ručne v `.env` — bez neho `rc=2` s odkazom na varovanie v `compose.sandbox-socket.yaml`
   (pripojenie hostiteľského docker socketu do workera je ekvivalent roota). `run-local.sh`
   pridá k tomuto súhlasu `-f compose.sandbox-socket.yaml` automaticky. `.env` s
   `UNFLATTEN_SANDBOX_ALLOW_ROOTFUL=1` a bez `UNFLATTEN_ACCEPT_ROOTFUL_SOCKET=1` sa odmietne
   (`rc=2`) — dve cesty k tomu istému súhlasu by boli mätúce;
4. `deploy/docker/models.json` chýba alebo nevaliduje (`load_models_config`) → `rc=2` s
   odkazom na `dev-bundle.sh`;
5. `docker compose … build`;
6. `docker compose … up -d`, vypíše URL (loopback väzba: sprievodca pýta len heslo) alebo pri
   neloopback `UNFLATTEN_BIND_ADDRESS` jednoklikový odkaz `…/setup#token=<t>` (vlastník
   2026-09-29, DEC-052 Dodatok A; podrobnosti RUNBOOK §10).

`down` nikdy nevolá `docker compose down -v` — dátový zväzok/bind mount ostáva netknutý.

## `dev-bundle.sh` — lokálne varianty modelu pre docker compose

```bash
deploy/local/dev-bundle.sh                                    # predvolený checkpoint
deploy/local/dev-bundle.sh --checkpoint /iná/cesta/checkpoint.pt --run-dir /iná/cesta
```

Postaví (alebo znovupoužije) každú z kombinácií `bf16/bf16`, `int8/bf16`, `fp8/fp8`,
`int4/fp8` pod `deploy/docker/models/<precision>-<kv_cache>/` (rovnaký hostiteľský adresár,
aký `docker-compose.yaml` mountuje na `/models` cez `UNFLATTEN_HOST_MODEL_BUNDLE_DIR`,
predvolene `./models`), a zapíše `deploy/docker/models.json` s poľom `bundle` nastaveným na
CESTU V KONTAJNERI (`/models/<precision>-<kv_cache>`, nie hostiteľskú cestu — worker
`Path(variant["bundle"])` číta priamo vo vlastnom súborovom systéme). Variant, ktorý sa
nepodarí postaviť, sa preskočí s varovaním; skript zlyhá iba vtedy, keď sa nepodarí ani jeden
(nie je z čoho vybrať `default_variant`). Rovnaký `--run-dir` blokátor ako nižšie (komponentová
identita, tokenizer hash) platí aj tu.

## `dev-native.sh` — natívny spúšťač (bez Dockeru)

`deploy/local/dev-native.sh` je jednopríkazový obal nad `deploy/service/service.sh`
pre vlastníka na jeho vlastnom počítači s jednou GPU (owner requirement 2026-09-27,
zovšeobecnené: "sprav to tak, že to nebude len pre RTX 4070, ale všeobecne"). Nepridáva
žiadnu novú bezpečnostnú výnimku okrem toho, čo je tu výslovne vysvetlené nižšie.

```bash
deploy/local/dev-native.sh start    # vytvorí dáta, principal, sandbox, model bundle, web build, spustí API+worker
deploy/local/dev-native.sh status   # PID + /health/ready
deploy/local/dev-native.sh stop     # SIGTERM oboch procesov
deploy/local/dev-native.sh verify   # to isté ako start, ale nič nespúšťa
```

## Čo `dev-native.sh` spraví, v poradí

0. **Web SPA**: ak `src/unflatten/service/ui/dist` neexistuje, spustí
   `npm --prefix web ci && npm --prefix web run build` a skopíruje `web/dist` ta (T47, plán
   note 2) — inak sa použije už postavený build.
1. **Dátový adresár**: `$HOME/unflatten-service-local` (premenná `UNFLATTEN_LOCAL_DATA_DIR`),
   `0700`, presne ako `service.sh` sám vytvára `UNFLATTEN_DATA_DIR`.
2. **Principal**: ak `$UNFLATTEN_LOCAL_DATA_DIR/principal.json` neexistuje, vygeneruje jeden
   náhodný bearer token (64 hex znakov, `openssl rand -hex 32`) so VŠETKÝMI oprávneniami
   (`submit_diagnostic`, `read_status`, `download_prediction_ir`, `cancel_own_job`,
   `download_docx`), zapíše ho na `0400` a **vypíše token na stdout** — to je presne to, čo
   treba vložiť do UI alebo do `curl -H "Authorization: Bearer …"`. Pri opätovnom spustení sa
   token nevytvára nanovo, iba sa znova vypíše z už existujúceho súboru.
3. **Sandbox obraz**: použije `unflatten/pdf-sandbox:local`, ak existuje; inak skúsi
   `unflatten/pdf-sandbox:v7-poppler24.02` (ak bol postavený skôr); inak ho postaví z
   `deploy/pdf-sandbox/Containerfile` (RUNBOOK §10).
4. **Presnosť (`--precision`)**: predvolené `auto` — zmeria SKUTOČNÚ voľnú VRAM
   (`torch.cuda.mem_get_info`) a compute capability tejto GPU a vyrieši ich cez
   `unflatten.model.quantization.auto_policy` (tabuľka
   `src/unflatten/service/policies/precision_auto_policy.v1.json`). Iba jeden riadok tejto
   tabuľky (RTX 4070 SUPER, `docs/runs/quantization-rtx4070-2026-09-27.md`) je **nameraný**;
   ostatné úrovne sú z neho **odvodené** s bezpečnostnou rezervou — pozri
   `docs/runs/local-service-2026-09-27.md` za presné čísla tejto konkrétnej karty.
   FP8 váhy/FP8 KV cache sa ponúkajú iba pri compute capability ≥ 8.9 (Ada/Hopper) — na tejto
   RTX 4070 SUPER (SM 8.9) preto `auto` pri dostatku voľnej VRAM vyrieši `fp8` s FP8 KV cache
   ako predvolenú, najkvalitnejšiu variantu, ktorá sa zmestí (vlastník ju bude používať ako
   prvú). Staršie karty (< SM 8.9) dostanú INT8/INT4 s bf16 KV cache. Ak nič nesedí do voľnej
   VRAM, `auto` **odmietne**
   s jasnou správou namiesto tichého OOM neskôr. `UNFLATTEN_PRECISION=int8` (atď.) vynúti
   konkrétnu hodnotu priamo, bez merania.
5. **Model bundle**: postaví (alebo znovupoužije, ak už existuje)
   `$UNFLATTEN_LOCAL_DATA_DIR/model-bundles/v15b-<precision>-<kv_cache>` z LOKÁLNEHO
   checkpointu (`$UNFLATTEN_CHECKPOINT`, predvolene
   `~/unflatten-runs/ckpt-v15b-140000/checkpoint.pt` — krok 140 000, sha256 overený proti
   produkcii pri kopírovaní; nastav `UNFLATTEN_CHECKPOINT=/iná/cesta/checkpoint.pt` pre iný
   checkpoint). **Nikdy nekopíruje hotový balík z
   produkcie** — identita tokenizéra (`tokenizer_artifact_set_hash`) sa počíta zo súborov v
   lokálnej HF cache a tá sa reálne líši od produkčnej (pamäť
   `tokenizer-hash-counts-ignored-names`); skopírovaný produkčný balík by na tomto stroji
   neprešiel `verify_bundle`.
6. **Spustenie**: `service.sh start` s `UNFLATTEN_HOST=127.0.0.1`,
   `UNFLATTEN_PRECISION`/`UNFLATTEN_KV_CACHE` na vyriešenú hodnotu z kroku 4.

## Známy blokátor: `--run-dir` musí niesť balík komponentovej identity

`scripts/build_model_bundle.py build` potrebuje popri checkpointe aj
`component-lock.json`, `qwen_text_allowlist.json`, `loop_guard_v2_caps.json` a
`qwen-pinned-hashes.json` (predvolene z `--run-dir`, tu odvodené z priečinka checkpointu).
Na tomto stroji `~/unflatten-runs/ckpt-v15b-latest/` obsahuje IBA `checkpoint.pt` a jeho
ancestry sidecar — žiadny z týchto štyroch súborov tam nie je, lebo checkpoint bol
natrénovaný a jeho identitu naposledy overovali na produkčnom hostiteľovi, ktorý si tieto
súbory drží vo vlastnom `runs/s4-v15b/`. Bez nich `dev-native.sh start`/`verify` (aj
`dev-bundle.sh`) zlyhá s jasnou správou pri kroku "building local model bundle" — to je
OČAKÁVANÉ, nie chyba tohto skriptu.

Navyše aj keby si tieto štyri súbory odniekiaľ skopíroval: checkpoint má v sebe zapečatený
`cache_identity.tokenizer_artifact_set_hash` vypočítaný z PRODUKČNEJ HF cache. Lokálna HF
cache (rovnaký model, rovnaká revízia, ale iný súbor-po-súbore výpis adresára) dáva INÝ hash
(pozri `docs/runs/local-service-2026-09-27.md`) — `load_candidate_runtime`'s tokenizer gate
preto odmietne aj korektne postavený balík. Toto NIE JE niečo, čo by tento skript (alebo
akýkoľvek iný kód v tejto session) smel obísť — je to skutočná bezpečnostná kontrola
(identita tokenizéra), ktorá tu správne odmieta nezhodu. Riešenie je na vlastníkovi:
buď preniesť presne tú istú HF snapshot štruktúru z produkcie, alebo nechať checkpoint
prejsť novým S1/S4 behom, ktorého identita bude počítaná z tejto lokálnej cache.

## Prečo `UNFLATTEN_SANDBOX_ALLOW_ROOTFUL=1` iba tu (`dev-native.sh`)

Docker na tomto stroji beží rootful (bežná desktopová inštalácia). RUNBOOK §10 aj
`deploy/service/service.env.example` majú tento prepínač predvolene vypnutý — je to
vlastníkovo rozhodnutie pre JEDNU pracovnú stanicu, ktorú používa iba on sám, nikdy
predvolená hodnota v zdieľanom `service.sh`/`service.env.example`. Iné nasadenie musí
o tomto rozhodnúť samostatne (rootless docker/podman je odporúčaný predvolený spôsob).
Pre `run-local.sh` (docker compose) je rovnaký súhlas explicitný cez
`UNFLATTEN_ACCEPT_ROOTFUL_SOCKET=1` v `.env` (viď vyššie) — nie tento premenný názov.

## Reálne meranie

`docs/runs/local-service-2026-09-27.md` obsahuje namerané VRAM/časy pre kvantizovanú
cestu načítania (`_move_candidate_to_device`, `quantize_candidate`, FP8 KV cache) na tejto
skutočnej RTX 4070 SUPER so skutočným v15b checkpointom, a presný opis vyššie uvedeného
blokátora pre plný HTTP beh cez `service.sh`.

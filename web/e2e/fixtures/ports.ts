/**
 * T41: single source of truth for the backend ports `playwright.config.ts`
 * spins up (see that file's own comment for the full rationale) -- spec
 * files that need the shared `app`/`session-expiry` backend import from
 * here instead of duplicating the port number.
 *
 * `UNFLATTEN_E2E_PORT_OFFSET` (default 0) shifts every backend port, so a
 * run can avoid a server someone else is using on the default ports (e.g. a
 * UI preview on 8431) instead of silently reusing it
 * (`reuseExistingServer`). Run with `CI=1` as well to refuse reuse outright.
 */
const OFFSET = Number(process.env.UNFLATTEN_E2E_PORT_OFFSET ?? '0')

function perProject(chromium: number, firefox: number, webkit: number) {
  return { chromium: chromium + OFFSET, firefox: firefox + OFFSET, webkit: webkit + OFFSET } as const
}

export const WIZARD_PORTS = perProject(8431, 8432, 8433)
// Owner 2026-09-29 (DEC-052 addendum): `WIZARD_PORTS` backends run with
// `UNFLATTEN_SETUP_TOKENLESS=1` (the loopback-only password-only wizard);
// these run WITHOUT it -- the non-loopback shape, where setup needs the
// `/setup#token=` link. One per project, for the same one-shot reason.
export const SETUP_LINK_PORTS = perProject(8434, 8435, 8436)
export const APP_PORT = 8440 + OFFSET
export const SESSION_EXPIRY_PORT = 8450 + OFFSET
// T46: one dedicated backend per project (like `WIZARD_PORTS`, not shared
// like `APP_PORT`) -- `models.spec.ts` mutates the same fixed three variant
// ids (download/load/set-default/resident state), so two projects hitting
// one shared backend concurrently would race each other's state, unlike the
// `app`/`session-expiry` backends whose specs only ever touch their own
// uniquely-labelled job.
export const MODELS_PORTS = perProject(8460, 8461, 8462)

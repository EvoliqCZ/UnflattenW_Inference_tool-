import { cleanup } from '@testing-library/react'
import { afterAll, afterEach } from 'vitest'
import '@testing-library/jest-dom/vitest'
import { server } from './msw/server'

// `@testing-library/react`'s own auto-cleanup only self-registers when it
// finds `afterEach` on the global scope; `vitest.config.ts` sets
// `test.globals: false` (test files import `describe`/`it`/... explicitly),
// so it never fires and DOM from one test leaks into the next -- most
// visibly as duplicate elements, or Radix's `pointer-events: none` body lock
// from an unclosed dialog/dropdown blocking the next test's clicks.
afterEach(() => {
  cleanup()
})

// The API client uses a same-origin, relative `baseUrl: ""` (design §7.4) --
// correct in a real browser, which resolves relative fetches/`Request`s
// against `window.location`. Vitest's `fetch`/`Request` are Node's (undici),
// which have no document to resolve against and throw `Invalid URL` before
// MSW ever sees the request (`openapi-fetch` builds its own `new
// Request(path, init)`, so patching `fetch` alone is not enough). Test-only:
// resolve root-relative paths against jsdom's `window.location` (set to
// `http://localhost/` in `vitest.config.ts`) the way a browser would.
function resolveAgainstLocation(input: string): string {
  return input.startsWith('/') ? new URL(input, window.location.origin).toString() : input
}
// T26 -> T29 (design §7.5 note): react-router's data router builds its own
// `Request` for every loader run, carrying the `signal` of an `AbortController`
// it creates itself. That `AbortController`/`AbortSignal` comes from jsdom
// (this environment's `globalThis.AbortController`), while `Request` here
// stays Node's (undici) -- undici's `Request` brand-checks `init.signal` and
// throws `TypeError: RequestInit: Expected signal (...) to be an instance of
// AbortSignal` for a signal from any other realm. Real routes never see this
// (a browser's `Request`/`AbortController` are the same realm); it is a
// test-environment-only mismatch. Dropping the foreign signal is safe here:
// no test in this suite asserts that a loader request aborts, and retrying
// `super()` after its first (rejected) call is well-defined -- a throwing
// `super()` call never initializes `this`, so the derived constructor is
// free to call it again.
const NodeRequest = globalThis.Request
class LocationAwareRequest extends NodeRequest {
  constructor(input: RequestInfo | URL, init?: RequestInit) {
    const resolvedInput = typeof input === 'string' ? resolveAgainstLocation(input) : input
    try {
      super(resolvedInput, init)
    } catch (error) {
      if (init?.signal && error instanceof TypeError) {
        const initWithoutSignal: RequestInit = { ...init }
        delete initWithoutSignal.signal
        super(resolvedInput, initWithoutSignal)
      } else {
        throw error
      }
    }
  }
}
globalThis.Request = LocationAwareRequest

const nodeFetch = globalThis.fetch
globalThis.fetch = (input: RequestInfo | URL, init?: RequestInit) =>
  nodeFetch(typeof input === 'string' ? resolveAgainstLocation(input) : input, init)

// MSW (design §7.6): fail the test on any request the handlers don't cover.
// Started here at module scope, not in `beforeAll` -- `api/client.ts` builds
// its `openapi-fetch` client (and captures `globalThis.fetch`/`Request`,
// see above) the moment a test file imports it, which happens after this
// setup file has fully run but before any `beforeAll` callback fires; a
// `beforeAll(() => server.listen())` here would patch `fetch` too late for
// that captured reference ever to see it.
server.listen({ onUnhandledRequest: 'error' })
afterEach(() => {
  server.resetHandlers()
})
afterAll(() => {
  server.close()
})

// jsdom's implementations of these are missing or incomplete; several Radix
// primitives (Dialog, Select, ScrollArea, Popover, ...) call them
// unconditionally when they open or measure themselves. Real no-ops for a
// test environment, not swallowed errors -- disabled rule-by-rule with a
// reason rather than for the whole file.
window.HTMLElement.prototype.scrollIntoView = function scrollIntoViewStub() {
  /* jsdom has no layout engine; nothing to scroll to. */
}
// eslint-disable-next-line @typescript-eslint/no-unnecessary-type-assertion -- lib.dom types this as always present; jsdom does not implement it.
window.HTMLElement.prototype.hasPointerCapture = (() => false) as HTMLElement['hasPointerCapture']
window.HTMLElement.prototype.releasePointerCapture = function releasePointerCaptureStub() {
  /* no-op: see hasPointerCapture above. */
}

if (typeof window.ResizeObserver === 'undefined') {
  class ResizeObserverStub implements ResizeObserver {
    observe() {
      /* no layout in jsdom to observe. */
    }
    unobserve() {
      /* no layout in jsdom to observe. */
    }
    disconnect() {
      /* nothing was observed. */
    }
  }
  window.ResizeObserver = ResizeObserverStub
}

// lib.dom types `Window.matchMedia` as always present, so `tsc -b` narrows a
// `typeof window.matchMedia === 'undefined'` guard to `never` -- jsdom does
// not actually implement it, so this always overwrites it with a base stub.
// Tests that need a specific `matches` value stub it per-test with
// `vi.stubGlobal`, reset by `vi.unstubAllGlobals()` back to this.
// eslint-disable-next-line @typescript-eslint/no-unnecessary-type-assertion -- eslint's and `tsc -b`'s project resolution disagree on this inferred type; the cast is load-bearing for the build.
window.matchMedia = ((query: string) => ({
  matches: false,
  media: query,
  onchange: null,
  // eslint-disable-next-line @typescript-eslint/no-deprecated -- Radix compatibility requires the legacy MediaQueryList API.
  addListener: () => {
    /* deprecated MediaQueryList API, kept for older Radix internals. */
  },
  // eslint-disable-next-line @typescript-eslint/no-deprecated -- same legacy browser API surface.
  removeListener: () => {
    /* see addListener above. */
  },
  addEventListener: () => {
    /* tests that need change events stub this per-test with vi.stubGlobal. */
  },
  removeEventListener: () => {
    /* see addEventListener above. */
  },
  dispatchEvent: () => false,
})) as typeof window.matchMedia

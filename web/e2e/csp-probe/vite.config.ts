import { randomBytes } from 'node:crypto'
import { readFileSync } from 'node:fs'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

const CSP_NONCE_PLACEHOLDER = '__CSP_NONCE__'

/**
 * Permanent regression for the T08 finding (design §6.3, DEC-052 bullet 1):
 * emulates `src/unflatten/service/web_static.py`'s `load_dist`/`serve_index`
 * contract -- a fresh `style-src` nonce minted per request, substituted into
 * both the `<meta name="csp-nonce" content="__CSP_NONCE__">` placeholder and
 * the `Content-Security-Policy` response header. `script-src` stays `'self'`
 * with no nonce (no inline script anywhere). Runs unconditionally on
 * `vite preview` for this fixture only -- this is not the throwaway T08
 * probe's opt-in comparison harness, it is the one and only mode this
 * fixture serves.
 */
function cspNonceRegressionPlugin(): Plugin {
  return {
    name: 'csp-nonce-regression',
    configurePreviewServer(server) {
      server.middlewares.use((req, res, next) => {
        const path = req.url?.split('?')[0]
        if (path !== '/index.html' && path !== '/') {
          next()
          return
        }
        const nonce = randomBytes(16).toString('base64url')
        const distIndex = new URL('dist/index.html', import.meta.url)
        const html = readFileSync(distIndex, 'utf-8').replaceAll(CSP_NONCE_PLACEHOLDER, nonce)
        res.setHeader('Content-Type', 'text/html; charset=utf-8')
        res.setHeader(
          'Content-Security-Policy',
          "default-src 'none'; script-src 'self'; " +
            `style-src 'self' 'nonce-${nonce}'; img-src 'self'; font-src 'self'; ` +
            "connect-src 'self'; manifest-src 'self'; base-uri 'none'; " +
            "form-action 'self'; frame-ancestors 'none'; object-src 'none'",
        )
        res.end(html)
      })
    },
  }
}

export default defineConfig({
  root: import.meta.dirname,
  plugins: [react(), cspNonceRegressionPlugin()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})

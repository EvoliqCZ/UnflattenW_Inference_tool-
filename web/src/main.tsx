import { setNonce } from 'get-nonce'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './app'
import './styles/tokens.css'

// design §6.3 / DEC-052 bullet 1: `src/unflatten/service/web_static.py`
// (`load_dist`) requires the built index.html to carry the literal
// `<meta name="csp-nonce" content="__CSP_NONCE__">` placeholder, fails
// closed without it, and replaces the placeholder with a fresh per-request
// nonce in both the meta tag and the CSP header at serve time.
// react-style-singleton (used by Radix's react-remove-scroll) reads the
// resolved nonce via get-nonce's setNonce() so every <style> it injects
// carries it and satisfies `style-src 'self' 'nonce-<N>'`. Must run before
// the first render. No-op when the placeholder was never replaced (e.g.
// `vite dev` serving index.html directly, without web_static in front of it).
const CSP_NONCE_PLACEHOLDER = '__CSP_NONCE__'
const nonce = document.querySelector('meta[name="csp-nonce"]')?.getAttribute('content')
if (nonce && nonce !== CSP_NONCE_PLACEHOLDER) {
  setNonce(nonce)
}

const rootElement = document.getElementById('root')
if (!rootElement) {
  throw new Error('main: #root element missing from index.html')
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

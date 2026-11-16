import { setNonce } from 'get-nonce'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ProbeApp } from './ProbeApp'
import './probe.css'

// Mirrors src/main.tsx's bootstrap exactly (design §6.3 / web_static.py
// load_dist contract): the placeholder is only ever real once the nonce
// middleware below (vite.config.ts `cspNonceProbePlugin`) has replaced it.
const CSP_NONCE_PLACEHOLDER = '__CSP_NONCE__'
const nonce = document.querySelector('meta[name="csp-nonce"]')?.getAttribute('content')
const resolvedNonce = nonce && nonce !== CSP_NONCE_PLACEHOLDER ? nonce : undefined
if (resolvedNonce) {
  setNonce(resolvedNonce)
}

const rootElement = document.getElementById('root')
if (!rootElement) {
  throw new Error('csp-probe: #root element missing')
}

createRoot(rootElement).render(
  <StrictMode>
    <ProbeApp nonce={resolvedNonce} />
  </StrictMode>,
)

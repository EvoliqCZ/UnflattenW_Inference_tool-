import { resolve } from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// design §7.2 / §6.4: no legacy polyfill, no injected modulePreload shim.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    modulePreload: { polyfill: false },
    // design §6.3: CSP is `img-src 'self'` -- a small image inlined as a
    // `data:` URI would be blocked, so every imported asset ships as a file.
    assetsInlineLimit: 0,
  },
  server: {
    proxy: {
      '/v1': { target: 'http://127.0.0.1:8420', changeOrigin: true },
      '/health': { target: 'http://127.0.0.1:8420', changeOrigin: true },
    },
  },
  resolve: {
    alias: {
      '@': resolve(import.meta.dirname, 'src'),
    },
  },
})

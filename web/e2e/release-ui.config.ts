import { resolve } from 'node:path'
import { defineConfig, devices } from '@playwright/test'

// Isolated layout checks against synthetic API responses; no GPU worker or user data.
export default defineConfig({
  testDir: '.',
  testMatch: 'release-ui.spec.ts',
  reporter: 'list',
  use: { ...devices['Desktop Chrome'], baseURL: 'http://127.0.0.1:8477', locale: 'en-GB' },
  webServer: {
    cwd: resolve(import.meta.dirname, '..'),
    command: 'npm run build && npx vite preview --host 127.0.0.1 --port 8477 --strictPort',
    url: 'http://127.0.0.1:8477',
    reuseExistingServer: false,
    timeout: 60_000,
  },
})

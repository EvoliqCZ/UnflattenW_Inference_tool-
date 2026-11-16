import { resolve } from 'node:path'
import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: '.',
  testMatch: 'api-documentation.spec.ts',
  reporter: 'list',
  use: { ...devices['Desktop Chrome'], baseURL: 'http://127.0.0.1:8478' },
  webServer: {
    cwd: resolve(import.meta.dirname, '../..'),
    command: 'PYTHONPATH=src python3 -m tests.service.api_docs_server',
    url: 'http://127.0.0.1:8478/health/live',
    reuseExistingServer: false,
    timeout: 30_000,
  },
})

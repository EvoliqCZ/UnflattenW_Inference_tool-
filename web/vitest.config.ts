import { resolve } from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': resolve(import.meta.dirname, 'src'),
    },
  },
  test: {
    environment: 'jsdom',
    // jsdom defaults to `about:blank`, which is not a valid base for the
    // relative URLs `openapi-fetch` builds (`baseUrl: ""`, design §7.4) --
    // MSW then never even sees the request (`fetch` throws before leaving
    // the process). A real origin lets the same relative paths resolve.
    environmentOptions: { jsdom: { url: 'http://localhost/' } },
    globals: false,
    setupFiles: ['./src/test/setup.ts'],
    exclude: ['node_modules', 'e2e', 'dist'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      include: ['src/**/*.{ts,tsx}'],
      // `components/ui/**` (T19) are thin shadcn/Radix wrappers -- vendored
      // primitives, not application logic (TDD rule "framework-generated
      // boilerplate" / "trivial ... with no logic"). They are exercised
      // through the features that use them starting T20; testing them here
      // would mostly re-test Radix itself.
      exclude: ['src/main.tsx', 'src/api/schema.d.ts', 'src/test/**', 'src/components/ui/**'],
      thresholds: {
        lines: 80,
      },
    },
  },
})

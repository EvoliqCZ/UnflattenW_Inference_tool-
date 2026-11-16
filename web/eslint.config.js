import js from '@eslint/js'
import reactHooks from 'eslint-plugin-react-hooks'
import globals from 'globals'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  { ignores: ['dist', 'coverage', 'playwright-report', 'test-results', 'src/api/schema.d.ts', 'e2e/csp-probe/dist'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.strictTypeChecked, ...tseslint.configs.stylisticTypeChecked],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    plugins: {
      'react-hooks': reactHooks,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'no-empty': ['error', { allowEmptyCatch: false }],
    },
  },
  {
    // Plain-JS build scripts, config files, e2e specs, and the vitest test for
    // check-dist.mjs (a .mjs module has no declaration file, so TS sees `any`
    // across that import boundary -- not worth chasing with strictTypeChecked).
    files: [
      '**/*.mjs',
      'vite.config.ts',
      'vitest.config.ts',
      'playwright.config.ts',
      'e2e/**/*',
      'scripts/check-dist.test.ts',
    ],
    extends: [tseslint.configs.disableTypeChecked],
  },
)

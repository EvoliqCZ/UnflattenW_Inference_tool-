#!/usr/bin/env node
// design §6.4 build guard: four rules over a built `dist/` directory. Runs as
// `postbuild` (so it also runs inside the Docker build stage). Exported as
// `checkDist(distDir)` so `check-dist.test.ts` can run it directly against
// small fixture directories, and as a CLI (`node check-dist.mjs <dir>`) that
// exits non-zero on any violation.
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { extname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

// http://www.w3.org/ - SVG namespace attribute, never fetched.
// https://react.dev/errors/ - React's own production bundle embeds this as a
// plain string (decodes minified invariant error codes for humans); it is
// not a resource the page loads or a network call, and it reappears in any
// React 19 production build, so it cannot be treated as a violation here.
// https://reactrouter.com/ , https://github.com/ungap/ , https://locize.com ,
// https://www.i18next.com/ - T29: react-router and i18next/react-i18next
// embed doc-reference URLs in their own dev-guidance strings (e.g. "no
// HydrateFallback provided", a URLSearchParams-polyfill note, i18next's
// startup banner); same shape as the react.dev case above -- plain text in
// an error/log message, never a resource the page fetches.
// http://localhost - react-router's own fallback base for `new URL(...)`
// parsing when it has no real `window.location` to resolve against; a
// hardcoded placeholder string in its source, not a request target.
const ALLOWED_URL_PREFIXES = [
  'http://www.w3.org/',
  'https://react.dev/errors/',
  'https://reactrouter.com/',
  'https://github.com/ungap/',
  'https://www.i18next.com/',
]
// item 8: these two have no trailing separator to anchor a `startsWith`
// prefix match against, so a lookalike host (`https://locize.com.evil.tld`)
// would otherwise pass -- match them exactly instead.
// EVOLIQ is a user-clicked publisher/contact link, never a fetched asset.
const ALLOWED_URL_EXACT = [
  'https://locize.com',
  'http://localhost',
  'https://www.evoliq.cz/sk',
  'https://www.evoliq.cz/en',
]
const ASSET_HASH_RE = /-[a-zA-Z0-9_-]{8,}\./
// T43/D-B: the one static, hand-placed `public/assets/` file Vite copies
// verbatim (no content hash) instead of bundling -- the setup wizard's demo
// PDF, served at the fixed URL `/assets/sample.pdf` (ReadyStep.tsx).
const STATIC_ASSET_EXCEPTIONS = ['sample.pdf']

/** @returns {string[]} absolute paths of every file under `dir`, recursively. */
function listFiles(dir) {
  const out = []
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name)
    if (entry.isDirectory()) {
      out.push(...listFiles(full))
    } else {
      out.push(full)
    }
  }
  return out
}

/**
 * @param {string} distDir
 * @returns {string[]} human-readable violation descriptions; empty means clean.
 */
export function checkDist(distDir) {
  const violations = []
  const files = listFiles(distDir)

  for (const file of files) {
    const rel = relative(distDir, file)
    const ext = extname(file)

    if (ext === '.html') {
      const html = readFileSync(file, 'utf-8')
      // Rule 1: no inline <script> without src, no <style>, no on*= handlers, no style= attribute.
      if (/<script(?![^>]*\bsrc=)[^>]*>/i.test(html)) {
        violations.push(`${rel}: inline <script> without src`)
      }
      if (/<style[\s>]/i.test(html)) {
        violations.push(`${rel}: inline <style> element`)
      }
      if (/\son[a-z]+\s*=/i.test(html)) {
        violations.push(`${rel}: inline on*= event handler attribute`)
      }
      if (/\sstyle\s*=/i.test(html)) {
        violations.push(`${rel}: inline style= attribute`)
      }
    }

    if (ext === '.html' || ext === '.js' || ext === '.css') {
      const text = readFileSync(file, 'utf-8')
      // Rule 2: no http(s):// URLs except the SVG w3.org namespace.
      const urlMatches = text.match(/https?:\/\/[^\s"'()]+/g) ?? []
      for (const url of urlMatches) {
        const allowed = ALLOWED_URL_PREFIXES.some((prefix) => url.startsWith(prefix)) || ALLOWED_URL_EXACT.includes(url)
        if (!allowed) {
          violations.push(`${rel}: external URL ${url}`)
        }
      }
    }

    if (ext === '.css') {
      // Rule 3: no CSS url(data:...).
      const text = readFileSync(file, 'utf-8')
      if (/url\(\s*['"]?data:/i.test(text)) {
        violations.push(`${rel}: CSS url(data:...) found`)
      }
    }
  }

  // Rule 4: every file directly under dist/assets carries a content hash in its name.
  const assetsDir = join(distDir, 'assets')
  try {
    if (statSync(assetsDir).isDirectory()) {
      for (const file of listFiles(assetsDir)) {
        const rel = relative(distDir, file)
        if (!ASSET_HASH_RE.test(file) && !STATIC_ASSET_EXCEPTIONS.includes(relative(assetsDir, file))) {
          violations.push(`${rel}: asset file name has no content hash`)
        }
      }
    }
  } catch {
    // no assets/ directory (e.g. a tiny fixture) - nothing to check for rule 4.
  }

  // Rule 5 (UX H8): a shipped `theme-init.js` must be loaded by index.html as a
  // blocking script before the app bundle; otherwise it is dead code and the
  // pre-login screens never follow the saved or system theme.
  const indexPath = join(distDir, 'index.html')
  if (files.includes(join(distDir, 'theme-init.js')) && files.includes(indexPath)) {
    const html = readFileSync(indexPath, 'utf-8')
    const themeAt = html.search(/<script(?![^>]*\btype=["']module["'])[^>]*\bsrc=["']\/theme-init\.js["'][^>]*>/i)
    const moduleAt = html.search(/<script[^>]*\btype=["']module["']/i)
    if (themeAt < 0 || (moduleAt >= 0 && moduleAt < themeAt)) {
      violations.push('index.html: theme-init.js is not loaded as a blocking script before the app bundle')
    }
  }

  return violations
}

async function main() {
  const distDir = process.argv[2]
  if (!distDir) {
    console.error('usage: check-dist.mjs <dist-dir>')
    process.exit(2)
  }
  const violations = checkDist(distDir)
  if (violations.length > 0) {
    console.error(`check-dist: ${String(violations.length)} violation(s) in ${distDir}:`)
    for (const v of violations) {
      console.error(`  - ${v}`)
    }
    process.exit(1)
  }
  console.log(`check-dist: ${distDir} clean`)
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  await main()
}

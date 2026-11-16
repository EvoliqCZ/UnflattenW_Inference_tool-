// T-fix (2026-09-27): regression coverage for the e2e harness staleness bug
// (ensure-build.sh / start-api.sh silently reused an outdated dist/ build).
// Runs `lib-freshness.sh`'s `needs_build`/`needs_copy` as real bash against
// disposable temp trees -- proving the staleness *rule*, not re-reading the
// shell logic by eye.
import { execFileSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, rmSync, utimesSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

const LIB_PATH = join(import.meta.dirname, 'lib-freshness.sh')

let dir: string

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'freshness-'))
})

afterEach(() => {
  rmSync(dir, { recursive: true, force: true })
})

/** Runs `fn(...args)` from lib-freshness.sh; returns true (exit 0) / false (exit 1). */
function runPredicate(fn: string, args: string[]): boolean {
  try {
    execFileSync('bash', ['-c', 'source "$1"; f="$2"; shift 2; "$f" "$@"', 'bash', LIB_PATH, fn, ...args])
    return true
  } catch (err) {
    const status = (err as { status?: number }).status
    if (status === 1) return false
    throw err
  }
}

function touchOlder(path: string, secondsBeforeNow: number) {
  const t = new Date(Date.now() - secondsBeforeNow * 1000)
  utimesSync(path, t, t)
}

describe('needs_build', () => {
  it('is true when dist/index.html does not exist yet', () => {
    mkdirSync(join(dir, 'src'), { recursive: true })
    writeFileSync(join(dir, 'src', 'main.tsx'), 'x')

    expect(runPredicate('needs_build', [dir])).toBe(true)
  })

  it('is true when a tracked source file is newer than dist/index.html', () => {
    mkdirSync(join(dir, 'dist'), { recursive: true })
    mkdirSync(join(dir, 'src'), { recursive: true })
    const distIndex = join(dir, 'dist', 'index.html')
    const srcFile = join(dir, 'src', 'main.tsx')
    writeFileSync(distIndex, 'old build')
    writeFileSync(srcFile, 'x')
    touchOlder(distIndex, 100)
    // srcFile keeps its just-written (newer) mtime.

    expect(runPredicate('needs_build', [dir])).toBe(true)
  })

  it('is false when dist/index.html is newer than every tracked source file (nothing changed)', () => {
    mkdirSync(join(dir, 'dist'), { recursive: true })
    mkdirSync(join(dir, 'src'), { recursive: true })
    const srcFile = join(dir, 'src', 'main.tsx')
    const distIndex = join(dir, 'dist', 'index.html')
    writeFileSync(srcFile, 'x')
    touchOlder(srcFile, 100)
    writeFileSync(distIndex, 'fresh build')

    expect(runPredicate('needs_build', [dir])).toBe(false)
  })
})

describe('needs_copy', () => {
  it('is true when the target has no index.html yet', () => {
    const src = join(dir, 'src-dist')
    const dst = join(dir, 'dst-dist')
    mkdirSync(src, { recursive: true })
    writeFileSync(join(src, 'index.html'), 'built')

    expect(runPredicate('needs_copy', [src, dst])).toBe(true)
  })

  it('is true when the target content differs from a rebuilt source (the T45 staleness bug)', () => {
    const src = join(dir, 'src-dist')
    const dst = join(dir, 'dst-dist')
    mkdirSync(src, { recursive: true })
    mkdirSync(dst, { recursive: true })
    writeFileSync(join(src, 'index.html'), 'new build content')
    writeFileSync(join(dst, 'index.html'), 'old build content')

    expect(runPredicate('needs_copy', [src, dst])).toBe(true)
  })

  it('is false when the target already matches the source (no re-copy needed)', () => {
    const src = join(dir, 'src-dist')
    const dst = join(dir, 'dst-dist')
    mkdirSync(src, { recursive: true })
    mkdirSync(dst, { recursive: true })
    writeFileSync(join(src, 'index.html'), 'same content')
    writeFileSync(join(dst, 'index.html'), 'same content')

    expect(runPredicate('needs_copy', [src, dst])).toBe(false)
  })
})

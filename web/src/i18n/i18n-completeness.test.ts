import { describe, expect, it } from 'vitest'
import codesJson from '../../codes.json'
import { CODE_GROUPS } from './codes'
import en from './en.json'
import sk from './sk.json'

type JsonRecord = Record<string, unknown>

function getPath(root: JsonRecord, path: string): unknown {
  return path.split('.').reduce<unknown>((node, segment) => {
    if (typeof node !== 'object' || node === null) return undefined
    return (node as JsonRecord)[segment]
  }, root)
}

/** All leaf key paths in a nested translation object, e.g. `codes.state.queued`. */
function flattenKeys(node: unknown, prefix = ''): string[] {
  if (typeof node !== 'object' || node === null) {
    return prefix === '' ? [] : [prefix]
  }
  return Object.entries(node as JsonRecord).flatMap(([key, value]) =>
    flattenKeys(value, prefix === '' ? key : `${prefix}.${key}`),
  )
}

const codesGroups = codesJson as unknown as Record<string, unknown>
const groupNames = Object.keys(codesJson).filter((key) => key !== 'schema_version')

describe('i18n code catalogs', () => {
  it('CODE_GROUPS covers every group in codes.json, and no more', () => {
    expect(Object.keys(CODE_GROUPS).sort()).toEqual([...groupNames].sort())
  })

  it.each(groupNames)('codes.ts register for %s matches codes.json', (groupName) => {
    const group = CODE_GROUPS[groupName as keyof typeof CODE_GROUPS]
    const fromCodesJson = [...(codesGroups[groupName] as string[])].sort()
    const fromCodesTs = [...group.register].sort()
    expect(fromCodesTs).toEqual(fromCodesJson)
  })

  it.each(groupNames)('sk.json has exactly one entry per code for %s, no dead keys', (groupName) => {
    const group = CODE_GROUPS[groupName as keyof typeof CODE_GROUPS]
    const node = getPath(sk, group.prefix)
    expect(node, `sk.json is missing the "${group.prefix}" block`).toBeTypeOf('object')
    const actualCodes = Object.keys(node as JsonRecord).sort()
    const expectedCodes = [...group.register].sort()
    expect(actualCodes).toEqual(expectedCodes)
    if (group.shape !== null) {
      for (const code of group.register) {
        const entry = (node as JsonRecord)[code]
        expect(Object.keys(entry as JsonRecord).sort()).toEqual([...group.shape].sort())
      }
    }
  })

  it.each(groupNames)('en.json has exactly one entry per code for %s, no dead keys', (groupName) => {
    const group = CODE_GROUPS[groupName as keyof typeof CODE_GROUPS]
    const node = getPath(en, group.prefix)
    expect(node, `en.json is missing the "${group.prefix}" block`).toBeTypeOf('object')
    const actualCodes = Object.keys(node as JsonRecord).sort()
    const expectedCodes = [...group.register].sort()
    expect(actualCodes).toEqual(expectedCodes)
    if (group.shape !== null) {
      for (const code of group.register) {
        const entry = (node as JsonRecord)[code]
        expect(Object.keys(entry as JsonRecord).sort()).toEqual([...group.shape].sort())
      }
    }
  })
})

describe('i18n catalog parity', () => {
  it('en.json defines exactly the same keys as sk.json (design §7.5)', () => {
    expect(flattenKeys(en).sort()).toEqual(flattenKeys(sk).sort())
  })

  it('every catalog value is a non-empty string', () => {
    for (const [language, keys] of [
      ['sk', flattenKeys(sk)],
      ['en', flattenKeys(en)],
    ] as const) {
      for (const key of keys) {
        const value = getPath(language === 'sk' ? sk : en, key)
        expect(typeof value, `${language}.json "${key}"`).toBe('string')
        expect((value as string).length, `${language}.json "${key}" is empty`).toBeGreaterThan(0)
      }
    }
  })
})

it('preserves every translation interpolation placeholder in English', () => {
  for (const key of flattenKeys(sk)) {
    const placeholders = (value: unknown) =>
      String(value)
        .match(/{{.*?}}/g)
        ?.sort() ?? []
    expect(placeholders(getPath(en, key)), key).toEqual(placeholders(getPath(sk, key)))
  }
})

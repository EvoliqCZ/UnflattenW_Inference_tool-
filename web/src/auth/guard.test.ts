import { describe, expect, it } from 'vitest'
import { resolveGuardRedirect, sanitizeNextPath, SPA_ROUTES } from './guard'

describe('SPA_ROUTES', () => {
  it('is read from routes.json, not duplicated', () => {
    expect(SPA_ROUTES).toEqual(['/', '/setup', '/login', '/jobs', '/jobs/:jobId', '/models', '/settings'])
  })
})

describe('resolveGuardRedirect (UX §2.2 order)', () => {
  const notSetUp = { setupRequired: true, authenticated: false }
  const loggedOut = { setupRequired: false, authenticated: false }
  const loggedIn = { setupRequired: false, authenticated: true }

  it('sends to /setup when the password is not set, even for /login', () => {
    expect(resolveGuardRedirect('/login', notSetUp)).toBe('/setup')
    expect(resolveGuardRedirect('/jobs', notSetUp)).toBe('/setup')
  })

  it('does not redirect away from /setup itself when setup is required', () => {
    expect(resolveGuardRedirect('/setup', notSetUp)).toBeNull()
  })

  it('setup-required takes priority over the authentication check', () => {
    // A session could theoretically report authenticated=true while
    // setup_required=true (principal exists, web-auth.json missing) --
    // §2.2's order is setup first regardless.
    expect(resolveGuardRedirect('/jobs', { setupRequired: true, authenticated: true })).toBe('/setup')
  })

  it('sends to /login with the target as next when there is no valid session', () => {
    expect(resolveGuardRedirect('/jobs', loggedOut)).toBe('/login?next=%2Fjobs')
  })

  it('does not redirect away from /login itself when logged out', () => {
    expect(resolveGuardRedirect('/login', loggedOut)).toBeNull()
  })

  it('encodes a parametrized target path in next', () => {
    expect(resolveGuardRedirect('/jobs/abc-123', loggedOut)).toBe('/login?next=%2Fjobs%2Fabc-123')
  })

  it('sends an authenticated visitor of /login or /setup to / by default', () => {
    expect(resolveGuardRedirect('/login', loggedIn)).toBe('/')
    expect(resolveGuardRedirect('/setup', loggedIn)).toBe('/')
  })

  it('honors a valid next for an authenticated visitor of /login', () => {
    expect(resolveGuardRedirect('/login', loggedIn, '/jobs')).toBe('/jobs')
    expect(resolveGuardRedirect('/login', loggedIn, '/jobs/xyz')).toBe('/jobs/xyz')
  })

  it('never lands an authenticated visitor back on /login or /setup via next', () => {
    expect(resolveGuardRedirect('/login', loggedIn, '/setup')).toBe('/')
    expect(resolveGuardRedirect('/setup', loggedIn, '/login')).toBe('/')
  })

  it('falls back to / for an authenticated visitor of /login with an unsafe next', () => {
    expect(resolveGuardRedirect('/login', loggedIn, 'https://evil.example/')).toBe('/')
  })

  it('lets an authenticated visitor through to any other route', () => {
    expect(resolveGuardRedirect('/jobs', loggedIn)).toBeNull()
    expect(resolveGuardRedirect('/jobs/abc-123', loggedIn)).toBeNull()
    expect(resolveGuardRedirect('/', loggedIn)).toBeNull()
    expect(resolveGuardRedirect('/settings', loggedIn)).toBeNull()
  })
})

describe('sanitizeNextPath (open-redirect guard)', () => {
  it('accepts every literal spa_routes entry', () => {
    expect(sanitizeNextPath('/')).toBe('/')
    expect(sanitizeNextPath('/jobs')).toBe('/jobs')
    expect(sanitizeNextPath('/models')).toBe('/models')
    expect(sanitizeNextPath('/settings')).toBe('/settings')
  })

  it('accepts a parametrized route with a concrete id in place of :jobId', () => {
    expect(sanitizeNextPath('/jobs/abc-123')).toBe('/jobs/abc-123')
    expect(sanitizeNextPath('/jobs/00000000-0000-0000-0000-000000000000')).toBe(
      '/jobs/00000000-0000-0000-0000-000000000000',
    )
  })

  it('drops a query string or fragment before matching', () => {
    expect(sanitizeNextPath('/jobs?state=active')).toBe('/jobs')
    expect(sanitizeNextPath('/jobs/abc-123#tab=compare')).toBe('/jobs/abc-123')
  })

  it('falls back to / for null, undefined or empty input', () => {
    expect(sanitizeNextPath(null)).toBe('/')
    expect(sanitizeNextPath(undefined)).toBe('/')
    expect(sanitizeNextPath('')).toBe('/')
  })

  it('falls back to / for a path not in spa_routes', () => {
    expect(sanitizeNextPath('/not-a-real-route')).toBe('/')
    expect(sanitizeNextPath('/jobs/abc/extra')).toBe('/')
  })

  it('rejects absolute URLs', () => {
    expect(sanitizeNextPath('http://evil.example/jobs')).toBe('/')
    expect(sanitizeNextPath('https://evil.example/')).toBe('/')
    expect(sanitizeNextPath('javascript:alert(1)')).toBe('/')
  })

  it('rejects protocol-relative //host', () => {
    expect(sanitizeNextPath('//evil.example')).toBe('/')
    expect(sanitizeNextPath('//evil.example/jobs')).toBe('/')
  })

  it('rejects backslash tricks browsers normalize to //host', () => {
    expect(sanitizeNextPath('/\\evil.example')).toBe('/')
    expect(sanitizeNextPath('\\\\evil.example')).toBe('/')
  })

  it('rejects percent-encoded variants of //host', () => {
    expect(sanitizeNextPath('/%2F%2Fevil.example')).toBe('/')
    expect(sanitizeNextPath('%2F%2Fevil.example')).toBe('/')
  })

  it('rejects percent-encoded backslash variants', () => {
    expect(sanitizeNextPath('%5Cevil.example')).toBe('/')
    expect(sanitizeNextPath('/%5Cevil.example')).toBe('/')
  })

  it('rejects double-encoded //host', () => {
    expect(sanitizeNextPath('%252F%252Fevil.example')).toBe('/')
  })

  it('rejects a percent-encoded absolute URL', () => {
    expect(sanitizeNextPath('http%3A%2F%2Fevil.example')).toBe('/')
  })

  it('falls back to / on malformed percent-encoding instead of throwing', () => {
    expect(sanitizeNextPath('/jobs%')).toBe('/')
  })

  it('accepts a custom spa_routes list for the closed-list check', () => {
    expect(sanitizeNextPath('/only-here', ['/only-here'])).toBe('/only-here')
    expect(sanitizeNextPath('/jobs', ['/only-here'])).toBe('/')
  })
})

import routes from '../../routes.json'

/**
 * design §7.5 / UX §2.2: the closed list of paths a redirect may ever land
 * on. Read from the same generated `routes.json` the server allow-lists
 * (`SPA_ROUTES`), not duplicated here.
 */
export const SPA_ROUTES: readonly string[] = routes.spa_routes

function routeToPattern(route: string): RegExp {
  const segments = route
    .split('/')
    .map((segment) => (segment.startsWith(':') ? '[^/]+' : segment.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')))
  return new RegExp(`^${segments.join('/')}$`)
}

const SPA_ROUTE_PATTERNS = SPA_ROUTES.map(routeToPattern)

const DEFAULT_NEXT = '/'

/**
 * UX §2.1 "next=<cesta>" / design §7.5: accepts `next` only when it is
 * (after decoding) exactly one of `spa_routes` (with `:param` segments
 * matching any single path segment). This is the sole open-redirect guard,
 * and it is deliberately a whitelist match rather than a blocklist of "bad"
 * shapes (absolute URL, protocol-relative `//host`, backslash, ...): every
 * entry in `spa_routes` starts with exactly one `/` followed by a non-`/`
 * character (or is `/` itself), so none of those shapes can ever equal a
 * route pattern -- a new smuggling trick still has to first produce a value
 * that is *literally* one of `spa_routes`, which by construction is never
 * cross-origin. (Mutation-tested: a version of this function with an
 * explicit "reject cross-origin-looking strings" pre-check added no
 * coverage -- every case it caught, the whitelist match already caught.)
 */
export function sanitizeNextPath(
  rawNext: string | null | undefined,
  spaRoutes: readonly string[] = SPA_ROUTES,
): string {
  if (!rawNext) {
    return DEFAULT_NEXT
  }

  let candidate = rawNext
  try {
    // `next` may reach here already decoded once (e.g. via
    // `URLSearchParams.get`) or still raw (e.g. read from `location.search`
    // by hand); decode until stable so a double-encoded smuggling attempt
    // (`%252F%252Fevil.com`) is caught by the whitelist match below instead
    // of slipping through as an opaque string that never matches a route
    // pattern anyway. Capped so a pathological input cannot loop.
    for (let i = 0; i < 5; i += 1) {
      const decoded = decodeURIComponent(candidate)
      if (decoded === candidate) break
      candidate = decoded
    }
  } catch {
    return DEFAULT_NEXT
  }

  const pathOnly = candidate.split(/[?#]/)[0] ?? DEFAULT_NEXT
  const patterns = spaRoutes === SPA_ROUTES ? SPA_ROUTE_PATTERNS : spaRoutes.map(routeToPattern)
  const isKnownRoute = patterns.some((pattern) => pattern.test(pathOnly))
  return isKnownRoute ? pathOnly : DEFAULT_NEXT
}

export interface GuardSessionState {
  readonly setupRequired: boolean
  readonly authenticated: boolean
}

/**
 * UX §2.2 decision order, evaluated in this order and no other:
 *
 * 1. password not set yet -> `/setup` (even if the target was `/login`)
 * 2. no valid session -> `/login?next=<target>`
 * 3. valid session, target is `/setup` or `/login` -> `next` (sanitized) or `/`
 * 4. valid session, any other target -> stay (`null`, no redirect)
 *
 * Returns the path (with query string) to redirect to, or `null` to render
 * `pathname` as requested.
 */
export function resolveGuardRedirect(
  pathname: string,
  session: GuardSessionState,
  next?: string | null,
): string | null {
  if (session.setupRequired) {
    return pathname === '/setup' ? null : '/setup'
  }

  if (!session.authenticated) {
    if (pathname === '/login') {
      return null
    }
    return `/login?next=${encodeURIComponent(pathname)}`
  }

  if (pathname === '/setup' || pathname === '/login') {
    const landing = sanitizeNextPath(next)
    // A sanitized `next` of `/setup` or `/login` is technically a known
    // route, but landing an authenticated user back on either would just
    // re-trigger this same branch -- send them to `/` instead (UX §2.2
    // step 3's "cieľ" is meant to be the page they actually wanted).
    return landing === '/setup' || landing === '/login' ? DEFAULT_NEXT : landing
  }

  return null
}

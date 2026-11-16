import type { QueryClient } from '@tanstack/react-query'
import { createBrowserRouter, Outlet, redirect, type LoaderFunctionArgs, type RouteObject } from 'react-router'
import { queryKeys } from '@/api/queries'
import { ProblemError } from '@/api/problem'
import { fetchSessionInfo } from '@/auth/session'
import { resolveGuardRedirect } from '@/auth/guard'
import routesConfig from '../routes.json'
import { AppShell } from '@/shell/AppShell'
import { AuthShell } from '@/shell/AuthShell'
import { ConvertPage } from '@/features/convert/page'
import { JobDetailPage } from '@/features/job-detail/page'
import { JobsPage } from '@/features/jobs/page'
import { LoginPage } from '@/features/login/page'
import { ModelsPage } from '@/features/models/page'
import { SettingsPage } from '@/features/settings/page'
import { SetupPage } from '@/features/setup/page'

/**
 * design §7.1/§7.5: the router's path list is generated from `routes.json`
 * (the same file the server's `SPA_ROUTES` allow-list reads), never
 * hand-duplicated. Each entry maps to the `features/*\/page.tsx` component
 * for that screen (UX §2.1) and to the shell that wraps it (UX §2.3:
 * `AuthShell` for the two pre-session screens, `AppShell` for the rest).
 */
export const ROUTE_PAGES: Record<string, () => React.JSX.Element> = {
  '/': ConvertPage,
  '/setup': SetupPage,
  '/login': LoginPage,
  '/jobs': JobsPage,
  '/jobs/:jobId': JobDetailPage,
  '/models': ModelsPage,
  '/settings': SettingsPage,
}

const AUTH_SHELL_ROUTES: readonly string[] = ['/setup', '/login']

export const SPA_ROUTES: readonly string[] = routesConfig.spa_routes

function toRoute(path: string): RouteObject {
  const Component = ROUTE_PAGES[path]
  if (!Component) {
    // router.test.ts asserts `ROUTE_PAGES` covers every `routes.json` entry;
    // this only fires if that invariant is broken at runtime.
    throw new Error(`router: no page component registered for route "${path}"`)
  }
  return { path, Component }
}

/**
 * Flat `{ path, Component }` list, one per `routes.json` entry, with no
 * shell or loader attached -- this is what `router.test.ts` checks against
 * `routes.json` in both directions (every configured path has a route here,
 * and every route here is a configured path).
 */
export function buildRouteDefinitions(): RouteObject[] {
  return SPA_ROUTES.map(toRoute)
}

/**
 * design §7.5 / UX §2.2: runs on every navigation before any route renders.
 * Fetches (and caches, so the shell's own session query reuses it) the
 * current `SessionInfo` and throws a `redirect()` when the guard order in
 * `auth/guard.ts` says the requested path is not the one to render.
 *
 * An invalid *session cookie* is a `401` from `GET /v1/session` (design
 * §2.3): treated as "no valid session" here rather than as a loader error,
 * so a stale cookie sends the visitor to `/login` instead of an error page.
 * Any other failure (network, 5xx) is rethrown for the router's error
 * boundary -- it is not this guard's job to decide what an unreachable API
 * means for navigation.
 */
function createGuardLoader(queryClient: QueryClient) {
  return async ({ request }: LoaderFunctionArgs) => {
    const url = new URL(request.url)
    let session: { setup_required: boolean; authenticated: boolean }
    try {
      session = await queryClient.query({
        queryKey: queryKeys.session(),
        queryFn: fetchSessionInfo,
      })
    } catch (error) {
      if (error instanceof ProblemError && error.status === 401) {
        session = { setup_required: false, authenticated: false }
      } else {
        throw error
      }
    }

    const redirectTo = resolveGuardRedirect(
      url.pathname,
      { setupRequired: session.setup_required, authenticated: session.authenticated },
      url.searchParams.get('next'),
    )
    if (redirectTo) {
      // react-router's own redirect mechanism: a loader signals a redirect
      // by throwing the `Response` `redirect()` builds, not an `Error`.
      // eslint-disable-next-line @typescript-eslint/only-throw-error -- react-router API contract, not application error handling.
      throw redirect(redirectTo)
    }
    return null
  }
}

/**
 * Nested route tree with the two shells and the guard loader, decoupled
 * from which router implementation renders it: `app.tsx` feeds this to
 * `createBrowserRouter` (real navigation), while a test can feed the same
 * tree to `createMemoryRouter` (no `window.history`/network-layer coupling).
 */
export function buildRouteTree(queryClient: QueryClient): RouteObject[] {
  const [authPaths, appPaths] = SPA_ROUTES.reduce<[string[], string[]]>(
    ([auth, app], path) => (AUTH_SHELL_ROUTES.includes(path) ? [[...auth, path], app] : [auth, [...app, path]]),
    [[], []],
  )

  return [
    {
      id: 'root',
      loader: createGuardLoader(queryClient),
      children: [
        {
          element: (
            <AuthShell>
              <Outlet />
            </AuthShell>
          ),
          children: authPaths.map(toRoute),
        },
        {
          element: (
            <AppShell>
              <Outlet />
            </AppShell>
          ),
          children: appPaths.map(toRoute),
        },
      ],
    },
  ]
}

/** Full router used by `app.tsx`: guard loader at the root, then one layout route per shell. */
export function createAppRouter(queryClient: QueryClient) {
  return createBrowserRouter(buildRouteTree(queryClient))
}

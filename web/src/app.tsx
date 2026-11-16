import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider } from 'react-router'
import { createQueryClient } from '@/api/queryClient'
import { SessionExpiredDialog } from '@/auth/SessionExpiredDialog'
import '@/i18n'
import { createAppRouter } from './router'

// design §7.1/§7.4/§7.5: one `QueryClient` and one router for the whole app.
// `createBrowserRouter` subscribes to `window.history` internally, and
// react-router's own guidance is to build it outside the component tree so
// remounts (e.g. React 19 `StrictMode`'s double-invoke) never leave a
// second, orphaned subscription running -- hence module-scope defaults
// rather than `useState(createAppRouter)`. `App` still takes both as
// (optional) props, mirroring `router.tsx`'s own `buildRouteTree(queryClient)`
// injection, so `app.test.tsx` can give each test a fresh, isolated pair
// instead of sharing the one real app instance's session cache.
const defaultQueryClient = createQueryClient()
const defaultRouter = createAppRouter(defaultQueryClient)

export interface AppProps {
  queryClient?: QueryClient
  router?: ReturnType<typeof createAppRouter>
}

/**
 * `SessionExpiredDialog` is mounted once, above the router, so a 401 on any
 * screen's query can open it (UX §3.2 point 5) without the dialog itself
 * living inside whichever route happened to trigger it.
 */
export function App({ queryClient = defaultQueryClient, router = defaultRouter }: AppProps = {}) {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
      <SessionExpiredDialog />
    </QueryClientProvider>
  )
}

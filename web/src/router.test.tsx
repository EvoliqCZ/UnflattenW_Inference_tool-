import { describe, expect, it } from 'vitest'
import routesConfig from '../routes.json'
import { buildRouteDefinitions, ROUTE_PAGES, SPA_ROUTES } from './router'

describe('router <-> routes.json parity', () => {
  it('reads SPA_ROUTES from routes.json, not a hand copy', () => {
    expect(SPA_ROUTES).toEqual(routesConfig.spa_routes)
  })

  it('every routes.json path has a registered page component', () => {
    for (const path of routesConfig.spa_routes) {
      expect(ROUTE_PAGES[path], `missing ROUTE_PAGES entry for "${path}"`).toBeTypeOf('function')
    }
  })

  it('ROUTE_PAGES has no entries outside routes.json (no orphan route)', () => {
    const configured = new Set(routesConfig.spa_routes)
    for (const path of Object.keys(ROUTE_PAGES)) {
      expect(configured.has(path), `ROUTE_PAGES has "${path}", which is not in routes.json`).toBe(true)
    }
  })

  it('buildRouteDefinitions produces exactly the routes.json path list, in order', () => {
    const paths = buildRouteDefinitions().map((route) => route.path)
    expect(paths).toEqual(routesConfig.spa_routes)
  })

  it('every built route has a Component', () => {
    for (const route of buildRouteDefinitions()) {
      expect(route.Component).toBeTypeOf('function')
    }
  })
})

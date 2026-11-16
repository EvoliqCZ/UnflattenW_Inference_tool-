import { setupServer } from 'msw/node'
import { handlers } from './handlers'

/** Node-side MSW server for Vitest (design §7.6); started in `test/setup.ts`. */
export const server = setupServer(...handlers)

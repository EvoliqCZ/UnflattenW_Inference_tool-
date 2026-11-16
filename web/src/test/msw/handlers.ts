import { createOpenApiHttp } from 'openapi-msw'
import type { paths } from '@/api/schema'
import {
  authenticatedSession,
  emptyJobList,
  modelCommandAccepted,
  modelsResponse,
  runningJob,
  settingsResponse,
} from './fixtures'

/**
 * design §7.6: handlers are typed from `schema.d.ts` via `openapi-msw`, so a
 * handler cannot return a shape the contract does not allow. Default,
 * happy-path handlers; tests override per-case with `server.use(...)`.
 */
const http = createOpenApiHttp<paths>()

export const handlers = [
  http.get('/v1/conversions/{job_id}', ({ response }) => response(200).json(runningJob)),
  // `state`/`limit`/`cursor` are read off the raw `Request` by `api.py` (not
  // declared `Query()` params -- see `features/jobs/api.ts`), so they carry
  // no typed `query` here either; the default is an empty list, and
  // `page.test.tsx` overrides this per case via `server.use(...)`.
  http.get('/v1/conversions', ({ response }) => response(200).json(emptyJobList)),
  http.delete('/v1/conversions/{job_id}', ({ response }) => response(200).json(runningJob)),
  http.get('/v1/models', ({ response }) => response(200).json(modelsResponse)),
  // Default: a valid session, so router-guard tests render the requested
  // route unless they override this with unauthenticatedSession/
  // setupRequiredSession via `server.use(...)`.
  http.get('/v1/session', ({ response }) => response(200).json(authenticatedSession)),
  http.get('/v1/settings', ({ response }) => response(200).json(settingsResponse)),
  http.post('/v1/models/{variant_id}/download', ({ response }) => response(202).json(modelCommandAccepted)),
  http.delete('/v1/models/{variant_id}/download', ({ response }) => response(202).json(modelCommandAccepted)),
  http.post('/v1/models/{variant_id}/load', ({ response }) =>
    response(202).json({ ...modelCommandAccepted, kind: 'load', phase: 'loading' }),
  ),
  http.put('/v1/models/default', ({ response }) => response(200).json(modelsResponse)),
]

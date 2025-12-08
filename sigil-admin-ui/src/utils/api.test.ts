import { beforeEach, describe, expect, it } from 'vitest'
import MockAdapter from 'axios-mock-adapter'
import type { AxiosRequestConfig } from 'axios'
import {
  apiClient,
  getStatus,
  getLogsTail,
  setApiKey,
} from '@/utils/api'

describe('api client utilities', () => {
  let mock: MockAdapter

  beforeEach(() => {
    mock = new MockAdapter(apiClient)
    mock.resetHistory()
    mock.resetHandlers()
    localStorage.clear()
  })

  it('includes admin API key header when stored', async () => {
    const payload = {
      admin: { host: '127.0.0.1', port: 8765, enabled: true },
      repos: {},
      index: { path: '/tmp/index', has_embeddings: true, embed_model: 'text-embedding-3' },
      watcher: { enabled: true, watching: [] },
    }
    setApiKey('secret-key')
    mock.onGet('/admin/status').reply((config: AxiosRequestConfig) => {
      expect(config.headers?.['X-Admin-Key']).toBe('secret-key')
      return [200, payload]
    })

    await expect(getStatus()).resolves.toEqual(payload)
  })

  it('surfaces structured error responses from server', async () => {
    const errorBody = { error: 'configuration_error', detail: 'admin_api_key_missing' }
    mock.onGet('/admin/status').reply(503, errorBody)

    await expect(getStatus()).rejects.toMatchObject(errorBody)
  })

  it('clamps log tail parameter to server limits', async () => {
    const logPayload = { path: '/tmp/logs/server.log', lines: ['first', 'second'] }
    mock.onGet('/admin/logs/tail').reply((config: AxiosRequestConfig) => {
      expect(config.params?.n).toBe(2000)
      return [200, logPayload]
    })

    await expect(getLogsTail(10_000)).resolves.toEqual(logPayload)
  })
})

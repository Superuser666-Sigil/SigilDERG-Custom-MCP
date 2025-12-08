import { describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { StatusPage } from './StatusPage'
import type { AdminStatusResponse, ErrorResponse } from '@/types/api'
import { getStatus } from '@/utils/api'

vi.mock('@/utils/api', () => ({
    getStatus: vi.fn(),
}))

const mockGetStatus = vi.mocked(getStatus)

describe('StatusPage', () => {
    beforeEach(() => {
        vi.clearAllMocks()
        vi.stubGlobal(
            'setInterval',
            vi.fn(() => 0 as unknown as ReturnType<typeof globalThis.setInterval>)
        )
    })

    afterEach(() => {
        vi.unstubAllGlobals()
    })

    it('renders loading state then displays status data', async () => {
        const payload: AdminStatusResponse = {
            admin: { host: '127.0.0.1', port: 8765, enabled: true },
            repos: { repoA: '/tmp/repoA' },
            index: { path: '/tmp/index', has_embeddings: true, embed_model: 'embedder' },
            watcher: { enabled: true, watching: ['repoA'] },
        }
        mockGetStatus.mockResolvedValueOnce(payload)

        render(<StatusPage />)

        await waitFor(() => expect(mockGetStatus).toHaveBeenCalled())
        expect(await screen.findByRole('heading', { name: /Server Status/i })).toBeInTheDocument()

        expect(screen.getByText('Admin API')).toBeInTheDocument()
        expect(screen.getByText('127.0.0.1')).toBeInTheDocument()
        expect(screen.getByText('8765')).toBeInTheDocument()
        expect(screen.getByText('embedder')).toBeInTheDocument()
        expect(screen.getByText('/tmp/repoA')).toBeInTheDocument()
    })

    it('displays error alert when fetch fails', async () => {
        const error: ErrorResponse = {
            error: 'configuration_error',
            detail: 'admin_api_key_missing',
        }
        mockGetStatus.mockRejectedValueOnce(error)

        render(<StatusPage />)

        await waitFor(() => expect(mockGetStatus).toHaveBeenCalled())
        expect(await screen.findByText('admin_api_key_missing')).toBeInTheDocument()
    })
})

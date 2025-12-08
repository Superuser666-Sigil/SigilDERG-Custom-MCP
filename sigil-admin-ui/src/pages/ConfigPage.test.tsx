import { describe, expect, it, vi, beforeEach } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { ConfigPage } from './ConfigPage'
import type { ConfigResponse, ErrorResponse } from '@/types/api'
import { getConfig } from '@/utils/api'

vi.mock('@/utils/api', () => ({
    getConfig: vi.fn(),
}))

const mockGetConfig = vi.mocked(getConfig)

const configResponse: ConfigResponse = {
    admin: { host: '127.0.0.1', port: 8000 },
    feature_flags: ['alpha'],
}

describe('ConfigPage', () => {
    beforeEach(() => {
        vi.clearAllMocks()
        mockGetConfig.mockResolvedValue(configResponse)
        vi.stubGlobal('navigator', {
            clipboard: {
                writeText: vi.fn().mockResolvedValue(undefined),
            },
        } as unknown as Navigator)
    })

    afterEach(() => {
        vi.unstubAllGlobals()
    })

    it('fetches and renders configuration JSON', async () => {
        render(<ConfigPage />)

        await waitFor(() => expect(mockGetConfig).toHaveBeenCalledTimes(1))
        expect(await screen.findByRole('heading', { level: 1, name: /Configuration/i })).toBeInTheDocument()
        expect(screen.getByText(/feature_flags/)).toBeInTheDocument()
    })

    it('refresh button refetches config', async () => {
        render(<ConfigPage />)
        await screen.findByText('Configuration')

        fireEvent.click(screen.getByRole('button', { name: /^Refresh$/i }))
        await waitFor(() => expect(mockGetConfig).toHaveBeenCalledTimes(2))
    })

    it('copy button writes JSON to clipboard', async () => {
        render(<ConfigPage />)
        await screen.findByText('Configuration')

        fireEvent.click(screen.getByRole('button', { name: /Copy JSON/i }))
        const clipboard = (navigator as Navigator).clipboard.writeText as ReturnType<typeof vi.fn>
        expect(clipboard).toHaveBeenCalledWith(JSON.stringify(configResponse, null, 2))
    })

    it('displays error alert when fetch fails', async () => {
        const error: ErrorResponse = { error: 'unauthorized', detail: 'bad key' }
        mockGetConfig.mockRejectedValueOnce(error)

        render(<ConfigPage />)

        expect(await screen.findByText('bad key')).toBeInTheDocument()
    })
})

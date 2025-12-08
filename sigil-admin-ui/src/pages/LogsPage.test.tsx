import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { LogsPage } from './LogsPage'
import type { LogsResponse, ErrorResponse } from '@/types/api'
import { getLogsTail } from '@/utils/api'

vi.mock('@/utils/api', () => ({
    getLogsTail: vi.fn(),
}))

const mockGetLogsTail = vi.mocked(getLogsTail)

const logsResponse: LogsResponse = {
    path: '/tmp/server.log',
    lines: ['line 1', 'line 2'],
}

describe('LogsPage', () => {
    beforeEach(() => {
        vi.clearAllMocks()
        mockGetLogsTail.mockResolvedValue(logsResponse)
        vi.stubGlobal('navigator', {
            clipboard: {
                writeText: vi.fn().mockResolvedValue(undefined),
            },
        } as unknown as Navigator)
    })

    afterEach(() => {
        vi.unstubAllGlobals()
    })

    it('fetches logs and renders entries', async () => {
        render(<LogsPage />)

        await waitFor(() => expect(mockGetLogsTail).toHaveBeenCalledWith(200))
        expect(screen.getByText('/tmp/server.log')).toBeInTheDocument()
        expect(screen.getByText('line 1')).toBeInTheDocument()
    })

    it('changes line count and triggers refetch', async () => {
        render(<LogsPage />)
        await waitFor(() => expect(mockGetLogsTail).toHaveBeenCalledTimes(1))

        const input = screen.getByRole('spinbutton')
        fireEvent.change(input, { target: { value: '50' } })

        await waitFor(() => expect(mockGetLogsTail).toHaveBeenCalledWith(50))
    })

    it('toggles auto-refresh and clears interval when paused', async () => {
        const intervalId = 12345 as unknown as ReturnType<typeof setInterval>
        const intervalSpy = vi.spyOn(globalThis, 'setInterval').mockReturnValue(intervalId)
        const clearIntervalSpy = vi.spyOn(globalThis, 'clearInterval')

        render(<LogsPage />)
        await waitFor(() => expect(mockGetLogsTail).toHaveBeenCalledTimes(1))

        fireEvent.click(screen.getByRole('button', { name: /Auto-refresh/i }))
        expect(intervalSpy).toHaveBeenCalledWith(expect.any(Function), 3000)

        fireEvent.click(screen.getByRole('button', { name: /Pause/i }))
        expect(clearIntervalSpy).toHaveBeenCalledWith(intervalId)

        intervalSpy.mockRestore()
        clearIntervalSpy.mockRestore()
    })

    it('copies logs to clipboard', async () => {
        render(<LogsPage />)
        await screen.findByText('Server Logs')

        fireEvent.click(screen.getByRole('button', { name: /^Copy$/i }))
        const clipboard = (navigator as Navigator).clipboard.writeText as ReturnType<typeof vi.fn>
        expect(clipboard).toHaveBeenCalledWith('line 1\nline 2')
    })

    it('shows error alert when fetch fails', async () => {
        const error: ErrorResponse = { error: 'timeout', detail: 'log fetch failed' }
        mockGetLogsTail.mockRejectedValueOnce(error)

        render(<LogsPage />)

        expect(await screen.findByText('log fetch failed')).toBeInTheDocument()
    })
})

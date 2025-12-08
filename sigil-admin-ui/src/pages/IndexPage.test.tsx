import { describe, expect, it, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { IndexPage } from './IndexPage'
import type { IndexStatsResponse, RebuildResponse, ErrorResponse } from '@/types/api'
import { getIndexStats, rebuildIndex } from '@/utils/api'

vi.mock('@/utils/api', () => ({
    getIndexStats: vi.fn(),
    rebuildIndex: vi.fn(),
}))

const mockGetIndexStats = vi.mocked(getIndexStats)
const mockRebuildIndex = vi.mocked(rebuildIndex)

const statsResponse: IndexStatsResponse = {
    total_documents: 1234,
    total_symbols: 5678,
    total_repos: 2,
    repos: {
        'repo-a': { documents: 100, symbols: 200, files: 10 },
        'repo-b': { documents: 200, symbols: 300, files: 20 },
    },
}

const rebuildResponse: RebuildResponse = {
    success: true,
    message: 'Rebuilt successfully',
    duration_seconds: 1.23,
}

describe('IndexPage', () => {
    beforeEach(() => {
        vi.clearAllMocks()
        mockGetIndexStats.mockResolvedValue(statsResponse)
        mockRebuildIndex.mockResolvedValue(rebuildResponse)
    })

    it('renders index statistics and repository table', async () => {
        render(<IndexPage />)

        await waitFor(() => expect(mockGetIndexStats).toHaveBeenCalledTimes(1))

        expect(screen.getByText('Index Management')).toBeInTheDocument()
        expect(screen.getAllByText('1,234')[0]).toBeInTheDocument()
        expect(screen.getAllByText('5,678')[0]).toBeInTheDocument()
        expect(screen.getAllByText('repo-a')[0]).toBeInTheDocument()
        expect(screen.getAllByText('repo-b')[0]).toBeInTheDocument()
    })

    it('opens rebuild dialog and submits rebuild request', async () => {
        render(<IndexPage />)
        await screen.findByText('Rebuild Index')

        fireEvent.click(screen.getByRole('button', { name: /Rebuild Index/i }))
        fireEvent.click(await screen.findByTestId('confirm-index-rebuild'))

        await waitFor(() =>
            expect(mockRebuildIndex).toHaveBeenCalledWith({
                repo: undefined,
                force: true,
            })
        )

        expect(await screen.findByText(/Rebuilt successfully/i)).toBeInTheDocument()
    })

    it('displays error alert when stats request fails', async () => {
        const error: ErrorResponse = { error: 'internal_error', detail: 'boom' }
        mockGetIndexStats.mockRejectedValueOnce(error)

        render(<IndexPage />)

        expect(await screen.findByText('boom')).toBeInTheDocument()
    })
})

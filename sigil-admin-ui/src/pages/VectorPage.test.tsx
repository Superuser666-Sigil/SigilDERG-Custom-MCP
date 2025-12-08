import { describe, expect, it, vi, beforeEach } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { VectorPage } from './VectorPage'
import type { AdminStatusResponse, ErrorResponse, RebuildResponse } from '@/types/api'
import { getStatus, rebuildVector } from '@/utils/api'

vi.mock('@/utils/api', () => ({
  getStatus: vi.fn(),
  rebuildVector: vi.fn(),
}))

const mockGetStatus = vi.mocked(getStatus)
const mockRebuildVector = vi.mocked(rebuildVector)

const statusResponse: AdminStatusResponse = {
  admin: { host: '127.0.0.1', port: 8000, enabled: true },
  repos: { repo1: '/path/repo1' },
  index: {
    path: '/tmp/vector.index',
    has_embeddings: true,
    embed_model: 'text-embed',
  },
  watcher: { enabled: false, watching: [] },
}

const rebuildSuccess: RebuildResponse = {
  success: true,
  message: 'Vector rebuild started',
  duration_seconds: 2.5,
}

describe('VectorPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockGetStatus.mockResolvedValue(statusResponse)
    mockRebuildVector.mockResolvedValue(rebuildSuccess)
  })

  it('loads status metadata and displays embeddings info', async () => {
    render(<VectorPage />)

    await waitFor(() => expect(mockGetStatus).toHaveBeenCalledTimes(1))
    expect(screen.getByText('Vector Index')).toBeInTheDocument()
    expect(screen.getByText('/tmp/vector.index')).toBeInTheDocument()
    expect(screen.getByText('text-embed')).toBeInTheDocument()
  })

  it('allows rebuilding vector index with repo/model selection', async () => {
    render(<VectorPage />)
    await screen.findByText('Rebuild Vector Index')

    fireEvent.click(screen.getByRole('button', { name: /Rebuild Vector Index/i }))
    const [repoSelect] = screen.getAllByRole('combobox')
    fireEvent.change(repoSelect, { target: { value: 'repo1' } })
    const modelInput = screen.getByPlaceholderText('text-embed')
    fireEvent.change(modelInput, { target: { value: 'custom-model' } })

    const confirmButton = await screen.findByTestId('confirm-vector-rebuild')
    fireEvent.click(confirmButton)

    await waitFor(() =>
      expect(mockRebuildVector).toHaveBeenCalledWith({
        repo: 'repo1',
        force: true,
        model: 'custom-model',
      })
    )

    expect(await screen.findByText(/Vector rebuild started/i)).toBeInTheDocument()
  })

  it('shows error alert when status fetch fails', async () => {
    const error: ErrorResponse = { error: 'down', detail: 'server offline' }
    mockGetStatus.mockRejectedValueOnce(error)

    render(<VectorPage />)

    expect(await screen.findByText('server offline')).toBeInTheDocument()
  })
})

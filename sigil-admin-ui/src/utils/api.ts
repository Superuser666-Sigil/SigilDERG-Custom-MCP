// Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial licenses are available. Contact: davetmire85@gmail.com

import axios, { AxiosError } from 'axios'
import type {
  AdminStatusResponse,
  IndexStatsResponse,
  RebuildResponse,
  LogsResponse,
  ConfigResponse,
  SaveConfigResponse,
  ErrorResponse,
  RebuildIndexRequest,
  RebuildVectorRequest,
} from '@/types/api'

export type {
  AdminStatusResponse,
  IndexStatsResponse,
  RebuildResponse,
  LogsResponse,
  ConfigResponse,
  SaveConfigResponse,
  ErrorResponse,
  RebuildIndexRequest,
  RebuildVectorRequest,
} from '@/types/api'

// Admin API is now integrated into main server (same process as MCP)
// Use main server port instead of separate admin port
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

// Get API key from localStorage or environment
const getApiKey = (): string | null => {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('admin_api_key')
  }
  return null
}

// Create axios instance with default config
export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add request interceptor for API key
apiClient.interceptors.request.use((config) => {
  const apiKey = getApiKey()
  if (apiKey) {
    config.headers['X-Admin-Key'] = apiKey
  }
  return config
})

// Error handling helper
const handleError = (error: unknown): ErrorResponse => {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<ErrorResponse>
    if (axiosError.response?.data) {
      return axiosError.response.data
    }
    return {
      error: 'network_error',
      detail: axiosError.message || 'Network request failed',
    }
  }
  return {
    error: 'unknown_error',
    detail: 'An unexpected error occurred',
  }
}

// API Functions

export const getStatus = async (): Promise<AdminStatusResponse> => {
  try {
    const response = await apiClient.get<AdminStatusResponse>('/admin/status')
    return response.data
  } catch (error) {
    throw handleError(error)
  }
}

export const getIndexStats = async (repo?: string): Promise<IndexStatsResponse> => {
  try {
    const params = repo ? { repo } : {}
    const response = await apiClient.get<IndexStatsResponse>('/admin/index/stats', { params })
    return response.data
  } catch (error) {
    throw handleError(error)
  }
}

export const rebuildIndex = async (request: RebuildIndexRequest = {}): Promise<RebuildResponse> => {
  try {
    const response = await apiClient.post<RebuildResponse>('/admin/index/rebuild', {
      repo: request.repo,
      force: request.force ?? true,
    })
    return response.data
  } catch (error) {
    throw handleError(error)
  }
}

export const rebuildVector = async (request: RebuildVectorRequest = {}): Promise<RebuildResponse> => {
  try {
    const response = await apiClient.post<RebuildResponse>('/admin/vector/rebuild', {
      repo: request.repo,
      force: request.force ?? true,
      model: request.model,
    })
    return response.data
  } catch (error) {
    throw handleError(error)
  }
}

export interface RebuildFileRequest {
  repo: string
  path: string
}

export const rebuildFile = async (request: RebuildFileRequest): Promise<RebuildResponse> => {
  try {
    const response = await apiClient.post<RebuildResponse>('/admin/index/file/rebuild', {
      repo: request.repo,
      path: request.path,
    })
    return response.data
  } catch (error) {
    throw handleError(error)
  }
}

export interface StaleIndexResponse {
  success: boolean
  repos: Record<string, {
    lance_db_path: string | null
    lance_db_size: number
    vectors_stale: number
    stale_documents: Array<{ path: string; error: string | null }>
  }>
}

export const getStaleIndex = async (repo?: string): Promise<StaleIndexResponse> => {
  try {
    const params = repo ? { repo } : {}
    const response = await apiClient.get<StaleIndexResponse>('/admin/index/stale', { params })
    return response.data
  } catch (error) {
    throw handleError(error)
  }
}

export const getLogsTail = async (n: number = 200): Promise<LogsResponse> => {
  try {
    const response = await apiClient.get<LogsResponse>('/admin/logs/tail', {
      params: { n: Math.max(1, Math.min(n, 2000)) },
    })
    return response.data
  } catch (error) {
    throw handleError(error)
  }
}

export const getConfig = async (): Promise<ConfigResponse> => {
  try {
    const response = await apiClient.get<ConfigResponse>('/admin/config')
    return response.data
  } catch (error) {
    throw handleError(error)
  }
}

export const saveConfig = async (config: ConfigResponse): Promise<SaveConfigResponse> => {
  try {
    const response = await apiClient.post<SaveConfigResponse>('/admin/config', config)
    return response.data
  } catch (error) {
    throw handleError(error)
  }
}

// Repo-level admin endpoints
export interface RepoInfoResponse {
  name: string
  path: string
  respect_gitignore: boolean
  ignore_patterns?: string[]
}

export const getRepoInfo = async (name: string): Promise<RepoInfoResponse> => {
  try {
    const response = await apiClient.get<RepoInfoResponse>(`/admin/repo/${encodeURIComponent(name)}`)
    return response.data
  } catch (error) {
    throw handleError(error)
  }
}

export const setRepoGitignore = async (name: string, respect: boolean, ignore_patterns?: string[]): Promise<RepoInfoResponse> => {
  try {
    const body: any = { respect_gitignore: respect }
    if (ignore_patterns !== undefined) body.ignore_patterns = ignore_patterns
    const response = await apiClient.post<RepoInfoResponse>(`/admin/repo/${encodeURIComponent(name)}/gitignore`, body)
    return response.data
  } catch (error) {
    throw handleError(error)
  }
}

// Helper to set API key
export const setApiKey = (key: string | null): void => {
  if (typeof window !== 'undefined') {
    if (key) {
      localStorage.setItem('admin_api_key', key)
    } else {
      localStorage.removeItem('admin_api_key')
    }
  }
}

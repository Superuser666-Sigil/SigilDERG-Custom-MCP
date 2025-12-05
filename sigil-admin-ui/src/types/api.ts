// Admin API Response Types

export interface AdminStatusResponse {
  admin: {
    host: string
    port: number
    enabled: boolean
  }
  repos: Record<string, string>
  index: {
    path: string
    has_embeddings: boolean
    embed_model: string | null
  }
  watcher: {
    enabled: boolean
    watching: string[]
  }
}

export interface IndexStatsResponse {
  total_documents: number
  total_symbols: number
  total_repos: number
  repos?: Record<string, {
    documents: number
    symbols: number
    files: number
  }>
}

export interface RebuildResponse {
  success: boolean
  repo?: string
  message?: string
  stats?: {
    documents: number
    symbols: number
    files: number
  }
  duration_seconds?: number
}

export interface LogsResponse {
  path: string
  lines: string[]
}

export interface ConfigResponse {
  [key: string]: any
}

export interface ErrorResponse {
  error: string
  detail?: string
  reason?: string
}

export interface RebuildIndexRequest {
  repo?: string
  force?: boolean
}

export interface RebuildVectorRequest {
  repo?: string
  force?: boolean
  model?: string
}



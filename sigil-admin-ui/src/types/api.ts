// Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial licenses are available. Contact: davetmire85@gmail.com

// Admin API Response Types

export interface AdminStatusResponse {
  admin: {
    host: string
    port: number
    enabled: boolean
    mode?: string
  }
  repos: Record<string, string>
  index: {
    path: string
    has_embeddings: boolean
    embed_model: string | null
    // New fields: whether embeddings are configured in config, and whether runtime embed function is ready
    embeddings_configured?: boolean
    embeddings_ready?: boolean
    trigram_backend?: string | null
    trigram_path?: string | null
    vector_backend?: string | null
    vector_path?: string | null
  }
  watcher: {
    enabled: boolean
    watching: string[]
  }
}

export interface IndexStatsResponse {
  total_documents: number
  total_symbols: number
  total_vectors?: number
  total_vectors_stale?: number
  total_repos: number
  repos?: Record<string, {
    documents: number
    symbols: number
    files: number
    vectors?: number
    vectors_stale?: number
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

export interface SaveConfigResponse {
  success: boolean
  path: string
  config: ConfigResponse
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

export interface RebuildFileRequest {
  repo: string
  path: string
}

export interface RepoInfoResponse {
  name: string
  path: string
  respect_gitignore: boolean
  ignore_patterns?: string[]
}


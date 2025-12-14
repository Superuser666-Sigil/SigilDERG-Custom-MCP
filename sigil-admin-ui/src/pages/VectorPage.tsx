// Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial licenses are available. Contact: davetmire85@gmail.com

import { useEffect, useState } from 'react'
import { rebuildVector, getStatus, getStaleIndex, rebuildFile } from '@/utils/api'
import type { ErrorResponse, RebuildResponse, AdminStatusResponse } from '@/types/api'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Select } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { ErrorAlert } from '@/components/ErrorAlert'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { RefreshCw, Sparkles } from 'lucide-react'
import { Badge } from '@/components/ui/badge'

export function VectorPage() {
  const [status, setStatus] = useState<AdminStatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<ErrorResponse | null>(null)
  const [selectedRepo, setSelectedRepo] = useState<string>('')
  const [model, setModel] = useState<string>('')
  const [rebuildDialogOpen, setRebuildDialogOpen] = useState(false)
  const [rebuilding, setRebuilding] = useState(false)
  const [rebuildResult, setRebuildResult] = useState<RebuildResponse | null>(null)
  const [staleInfo, setStaleInfo] = useState<Record<string, any> | null>(null)
  const [fileListRepo, setFileListRepo] = useState<string | null>(null)
  const [fileListOpen, setFileListOpen] = useState(false)
  const [filePage, setFilePage] = useState(0)
  const FILES_PER_PAGE = 20

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const data = await getStatus()
        setStatus(data)
        if (data.index.embed_model) {
          setModel(data.index.embed_model)
        }
        try {
          if (typeof getStaleIndex === 'function') {
            const stale = await getStaleIndex()
            setStaleInfo(stale.repos)
          }
        } catch (e) {
          // ignore stale fetch errors
        }
      } catch (err) {
        setError(err as ErrorResponse)
      } finally {
        setLoading(false)
      }
    }
    fetchStatus()
  }, [])

  const handleRebuild = async () => {
    try {
      setRebuilding(true)
      setError(null)
      setRebuildResult(null)
      const result = await rebuildVector({
        repo: selectedRepo || undefined,
        force: true,
        model: model || undefined,
      })
      setRebuildResult(result)
    } catch (err) {
      setError(err as ErrorResponse)
    } finally {
      setRebuilding(false)
    }
  }

  const repos = status?.repos ? Object.keys(status.repos) : []

  const formatSize = (bytes: number | null | undefined) => {
    const b = Number(bytes || 0)
    const mb = b / 1024 / 1024
    const gb = b / 1024 / 1024 / 1024
    if (mb < 1000) {
      return `${mb.toFixed(1)} MB` + (gb >= 0.1 ? ` (${gb.toFixed(2)} GB)` : '')
    }
    return `${gb.toFixed(2)} GB`
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Vector Index</h1>
          <p className="text-muted-foreground">Manage semantic search embeddings</p>
          {status && (
            <div className="flex flex-wrap gap-2 mt-2 items-center text-sm">
              {status.index.trigram_backend && (
                <Badge variant="outline">Trigram: {status.index.trigram_backend}</Badge>
              )}
              {status.index.vector_backend && (
                <Badge variant="outline">Vectors: {status.index.vector_backend}</Badge>
              )}
            </div>
          )}
        </div>
        <Button onClick={() => setRebuildDialogOpen(true)}>
          <Sparkles className="h-4 w-4 mr-2" />
          Rebuild Vector Index
        </Button>
      </div>

      {error && <ErrorAlert error={error} />}

      {loading ? (
        <div>Loading...</div>
      ) : status ? (
        <div className="grid gap-6 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5" />
                Embeddings Status
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Configured</span>
                <Badge variant={status.index.embeddings_configured ? 'default' : 'secondary'}>
                  {status.index.embeddings_configured ? 'Yes' : 'No'}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Runtime Ready</span>
                <Badge variant={status.index.embeddings_ready ? 'default' : 'secondary'}>
                  {status.index.embeddings_ready ? 'Ready' : 'Unavailable'}
                </Badge>
              </div>
              {status.index.embed_model && (
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Current Model</span>
                  <Badge variant="outline">{status.index.embed_model}</Badge>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Index Path</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground font-mono break-all">
                {status.index.path}
              </p>
            </CardContent>
          </Card>

          <Card className="md:col-span-2">
            <CardHeader>
              <CardTitle>Vector DBs</CardTitle>
              <CardDescription>Per-repository LanceDB location, size, and stale vector info</CardDescription>
            </CardHeader>
            <CardContent>
              {staleInfo ? (
                <div>
                  <table className="w-full table-auto">
                    <thead>
                      <tr>
                        <th className="text-left">Repository</th>
                        <th className="text-left">LanceDB Path</th>
                        <th className="text-right">Size</th>
                        <th className="text-right">Stale</th>
                        <th className="text-right">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(staleInfo).map(([repo, info]) => (
                        <tr key={repo} className="border-t">
                          <td className="py-2">{repo}</td>
                          <td className="py-2 font-mono break-all">{info.lance_db_path}</td>
                          <td className="py-2 text-right">{formatSize(info.lance_db_size)}</td>
                          <td className="py-2 text-right">
                            {info.vectors_stale > 0 ? (
                              <Badge variant="destructive">{info.vectors_stale}</Badge>
                            ) : (
                              <Badge variant="default">0</Badge>
                            )}
                          </td>
                          <td className="py-2 text-right">
                            <div className="text-sm">
                              <Button size="sm" onClick={() => {
                                setFileListRepo(repo)
                                setFilePage(0)
                                setFileListOpen(true)
                              }}>Show files</Button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div>Loading DB info...</div>
              )}
            </CardContent>
          </Card>
        </div>
      ) : null}

      <Dialog open={rebuildDialogOpen} onOpenChange={setRebuildDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rebuild Vector Index</DialogTitle>
            <DialogDescription>
              Rebuild the vector embeddings index for semantic search
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Repository (optional)</label>
              <Select
                value={selectedRepo}
                onChange={(e) => setSelectedRepo(e.target.value)}
              >
                <option value="">All Repositories</option>
                {repos.map((repo) => (
                  <option key={repo} value={repo}>
                    {repo}
                  </option>
                ))}
              </Select>
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Model (optional)</label>
              <Input
                value={model}
                onChange={(e) => setModel(e.target.value)}
                placeholder={status?.index.embed_model || 'Leave empty for default'}
              />
            </div>
            {rebuildResult && (
              <div className="space-y-2 p-4 bg-muted rounded-md">
                <div className="flex items-center gap-2">
                  <Badge variant={rebuildResult.success ? 'default' : 'destructive'}>
                    {rebuildResult.success ? 'Success' : 'Failed'}
                  </Badge>
                  {rebuildResult.duration_seconds && (
                    <span className="text-sm text-muted-foreground">
                      Completed in {rebuildResult.duration_seconds.toFixed(2)}s
                    </span>
                  )}
                </div>
                {rebuildResult.message && (
                  <p className="text-sm text-muted-foreground">{rebuildResult.message}</p>
                )}
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => {
              setRebuildDialogOpen(false)
              setRebuildResult(null)
            }}>
              Close
            </Button>
            <Button data-testid="confirm-vector-rebuild" onClick={handleRebuild} disabled={rebuilding}>
              {rebuilding ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Rebuilding...
                </>
              ) : (
                'Rebuild'
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      {/* File list dialog (paginated) */}
      <Dialog open={fileListOpen} onOpenChange={(v) => setFileListOpen(v)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Stale Files{fileListRepo ? ` — ${fileListRepo}` : ''}</DialogTitle>
            <DialogDescription>Paginated list of files with stale/missing vectors</DialogDescription>
          </DialogHeader>
          <div className="py-4">
            {fileListRepo && staleInfo && staleInfo[fileListRepo] ? (
              (() => {
                const docs = staleInfo[fileListRepo].stale_documents || []
                const start = filePage * FILES_PER_PAGE
                const pageDocs = docs.slice(start, start + FILES_PER_PAGE)
                return (
                  <div>
                    <div className="mb-2 text-sm text-muted-foreground">Showing {Math.min(docs.length, start + 1)}-{Math.min(docs.length, start + FILES_PER_PAGE)} of {docs.length}</div>
                    <ul className="space-y-2 text-sm">
                      {pageDocs.map((d: any) => (
                        <li key={d.path} className="flex items-center justify-between gap-2">
                          <div className="font-mono break-all">{d.path}</div>
                          <div className="flex items-center gap-2">
                            <div className="text-muted-foreground">{d.error ?? 'stale'}</div>
                            <Button size="sm" disabled={rebuilding} onClick={async () => {
                              try {
                                setRebuilding(true)
                                setError(null)
                                const resp = await rebuildFile({ repo: fileListRepo, path: d.path })
                                // If response indicates failure, surface it
                                if (!resp || (resp as any).success === false) {
                                  setError({ error: 'rebuild_failed', detail: (resp as any)?.message || 'Rebuild failed' })
                                } else {
                                  const fresh = await getStaleIndex()
                                  setStaleInfo(fresh.repos)
                                }
                              } catch (e) {
                                setError(e as ErrorResponse)
                              } finally {
                                setRebuilding(false)
                              }
                            }}>Reindex</Button>
                          </div>
                        </li>
                      ))}
                    </ul>
                    <div className="flex items-center justify-between mt-4">
                      <div>
                        <Button size="sm" variant="outline" onClick={() => setFilePage(Math.max(0, filePage - 1))} disabled={filePage === 0}>Prev</Button>
                        <Button size="sm" variant="outline" onClick={() => setFilePage(filePage + 1)} className="ml-2" disabled={(filePage + 1) * FILES_PER_PAGE >= (staleInfo[fileListRepo].stale_documents || []).length}>Next</Button>
                      </div>
                      <div className="text-sm text-muted-foreground">Page {filePage + 1}</div>
                    </div>
                  </div>
                )
              })()
            ) : (
              <div>No files</div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setFileListOpen(false)}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}


import { useEffect, useState } from 'react'
import { rebuildVector, type ErrorResponse, type RebuildResponse } from '@/utils/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Select } from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { ErrorAlert } from '@/components/ErrorAlert'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { RefreshCw, Sparkles } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { getStatus } from '@/utils/api'

export function VectorPage() {
  const [status, setStatus] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<ErrorResponse | null>(null)
  const [selectedRepo, setSelectedRepo] = useState<string>('')
  const [model, setModel] = useState<string>('')
  const [rebuildDialogOpen, setRebuildDialogOpen] = useState(false)
  const [rebuilding, setRebuilding] = useState(false)
  const [rebuildResult, setRebuildResult] = useState<RebuildResponse | null>(null)

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const data = await getStatus()
        setStatus(data)
        if (data.index.embed_model) {
          setModel(data.index.embed_model)
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

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Vector Index</h1>
          <p className="text-muted-foreground">Manage semantic search embeddings</p>
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
                <span className="text-sm font-medium">Enabled</span>
                <Badge variant={status.index.has_embeddings ? 'default' : 'secondary'}>
                  {status.index.has_embeddings ? 'Yes' : 'No'}
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
            <Button onClick={handleRebuild} disabled={rebuilding}>
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
    </div>
  )
}



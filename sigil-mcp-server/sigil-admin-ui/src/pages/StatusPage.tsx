import { useEffect, useState } from 'react'
import { getStatus, type AdminStatusResponse, type ErrorResponse } from '@/utils/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { LoadingSpinner } from '@/components/LoadingSpinner'
import { ErrorAlert } from '@/components/ErrorAlert'
import { StatusBadge } from '@/components/StatusBadge'
import { RefreshCw, Database, Eye, Server } from 'lucide-react'
import { Button } from '@/components/ui/button'

export function StatusPage() {
  const [status, setStatus] = useState<AdminStatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<ErrorResponse | null>(null)
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date())

  const fetchStatus = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await getStatus()
      setStatus(data)
      setLastRefresh(new Date())
    } catch (err) {
      setError(err as ErrorResponse)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 10000) // Refresh every 10 seconds
    return () => clearInterval(interval)
  }, [])

  if (loading && !status) {
    return <LoadingSpinner />
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Server Status</h1>
          <p className="text-muted-foreground">
            Last updated: {lastRefresh.toLocaleTimeString()}
          </p>
        </div>
        <Button onClick={fetchStatus} disabled={loading} variant="outline">
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {error && <ErrorAlert error={error} />}

      {status && (
        <div className="grid gap-6 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="h-5 w-5" />
                Admin API
              </CardTitle>
              <CardDescription>Admin API configuration and status</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Status</span>
                <StatusBadge status={status.admin.enabled} />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Host</span>
                <span className="text-sm text-muted-foreground">{status.admin.host}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Port</span>
                <span className="text-sm text-muted-foreground">{status.admin.port}</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                Index
              </CardTitle>
              <CardDescription>Index storage and embeddings configuration</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Path</span>
                <span className="text-sm text-muted-foreground font-mono text-xs truncate max-w-[200px]">
                  {status.index.path}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Embeddings</span>
                <StatusBadge status={status.index.has_embeddings} />
              </div>
              {status.index.embed_model && (
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Model</span>
                  <Badge variant="outline">{status.index.embed_model}</Badge>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Eye className="h-5 w-5" />
                File Watcher
              </CardTitle>
              <CardDescription>File system monitoring status</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Status</span>
                <StatusBadge status={status.watcher.enabled} />
              </div>
              <div>
                <span className="text-sm font-medium">Watching Repositories</span>
                <div className="mt-2 space-y-1">
                  {status.watcher.watching.length > 0 ? (
                    status.watcher.watching.map((repo) => (
                      <Badge key={repo} variant="secondary" className="mr-2">
                        {repo}
                      </Badge>
                    ))
                  ) : (
                    <span className="text-sm text-muted-foreground">None</span>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Repositories</CardTitle>
              <CardDescription>Configured repository paths</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {Object.entries(status.repos).map(([name, path]) => (
                  <div key={name} className="flex items-start justify-between border-b pb-2 last:border-0">
                    <div className="flex-1">
                      <div className="font-medium">{name}</div>
                      <div className="text-sm text-muted-foreground font-mono text-xs truncate">
                        {path}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}



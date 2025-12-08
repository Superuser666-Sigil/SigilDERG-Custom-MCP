// Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial licenses are available. Contact: davetmire85@gmail.com

import { useEffect, useState } from 'react'
import { getIndexStats, rebuildIndex } from '@/utils/api'
import type { IndexStatsResponse, ErrorResponse, RebuildResponse } from '@/types/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Select } from '@/components/ui/select'
import { LoadingSpinner } from '@/components/LoadingSpinner'
import { ErrorAlert } from '@/components/ErrorAlert'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { RefreshCw, Database, Play } from 'lucide-react'
import { Badge } from '@/components/ui/badge'

export function IndexPage() {
  const [stats, setStats] = useState<IndexStatsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<ErrorResponse | null>(null)
  const [selectedRepo, setSelectedRepo] = useState<string>('')
  const [rebuildDialogOpen, setRebuildDialogOpen] = useState(false)
  const [rebuilding, setRebuilding] = useState(false)
  const [rebuildResult, setRebuildResult] = useState<RebuildResponse | null>(null)

  const fetchStats = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await getIndexStats(selectedRepo || undefined)
      setStats(data)
    } catch (err) {
      setError(err as ErrorResponse)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStats()
  }, [selectedRepo])

  const handleRebuild = async () => {
    try {
      setRebuilding(true)
      setError(null)
      const result = await rebuildIndex({
        repo: selectedRepo || undefined,
        force: true,
      })
      setRebuildResult(result)
      await fetchStats() // Refresh stats after rebuild
    } catch (err) {
      setError(err as ErrorResponse)
    } finally {
      setRebuilding(false)
    }
  }

  const repos = stats?.repos ? Object.keys(stats.repos) : []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Index Management</h1>
          <p className="text-muted-foreground">Manage trigram and symbol indexes</p>
        </div>
        <div className="flex items-center gap-2">
          <Select
            value={selectedRepo}
            onChange={(e) => setSelectedRepo(e.target.value)}
            className="w-[200px]"
          >
            <option value="">All Repositories</option>
            {repos.map((repo) => (
              <option key={repo} value={repo}>
                {repo}
              </option>
            ))}
          </Select>
          <Button onClick={fetchStats} disabled={loading} variant="outline">
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button onClick={() => setRebuildDialogOpen(true)}>
            <Play className="h-4 w-4 mr-2" />
            Rebuild Index
          </Button>
        </div>
      </div>

      {error && <ErrorAlert error={error} />}

      {loading && !stats ? (
        <LoadingSpinner />
      ) : stats ? (
        <div className="grid gap-6 md:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                Total Documents
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{stats.total_documents.toLocaleString()}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Total Symbols</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{stats.total_symbols.toLocaleString()}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Repositories</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{stats.total_repos}</div>
            </CardContent>
          </Card>

          {stats.repos && Object.keys(stats.repos).length > 0 && (
            <Card className="md:col-span-3">
              <CardHeader>
                <CardTitle>Repository Breakdown</CardTitle>
                <CardDescription>Index statistics per repository</CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Repository</TableHead>
                      <TableHead>Documents</TableHead>
                      <TableHead>Symbols</TableHead>
                      <TableHead>Files</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {Object.entries(stats.repos).map(([repo, data]) => (
                      <TableRow key={repo}>
                        <TableCell className="font-medium">{repo}</TableCell>
                        <TableCell>{data.documents.toLocaleString()}</TableCell>
                        <TableCell>{data.symbols.toLocaleString()}</TableCell>
                        <TableCell>{data.files.toLocaleString()}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}
        </div>
      ) : null}

      <Dialog open={rebuildDialogOpen} onOpenChange={setRebuildDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rebuild Index</DialogTitle>
            <DialogDescription>
              {selectedRepo
                ? `Rebuild index for repository: ${selectedRepo}`
                : 'Rebuild index for all repositories'}
            </DialogDescription>
          </DialogHeader>
          {rebuildResult && (
            <div className="space-y-2">
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
              {rebuildResult.stats && (
                <div className="text-sm space-y-1">
                  <div>Documents: {rebuildResult.stats.documents.toLocaleString()}</div>
                  <div>Symbols: {rebuildResult.stats.symbols.toLocaleString()}</div>
                  <div>Files: {rebuildResult.stats.files.toLocaleString()}</div>
                </div>
              )}
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setRebuildDialogOpen(false)}>
              Cancel
            </Button>
            <Button data-testid="confirm-index-rebuild" onClick={handleRebuild} disabled={rebuilding}>
              {rebuilding ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Rebuilding...
                </>
              ) : (
                'Confirm Rebuild'
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}



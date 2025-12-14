// Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial licenses are available. Contact: davetmire85@gmail.com

import { useEffect, useState } from 'react'
import { getStatus, setApiKey, getRepoInfo, setRepoGitignore, type AdminStatusResponse, type ErrorResponse } from '@/utils/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { LoadingSpinner } from '@/components/LoadingSpinner'
import { ErrorAlert } from '@/components/ErrorAlert'
import { StatusBadge } from '@/components/StatusBadge'
import { RefreshCw, Database, Eye, Server } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'

const CODE_LANGUAGES: Array<{ id: string; label: string; exts: string[] }> = [
  { id: 'js_ts', label: 'JavaScript / TypeScript', exts: ['.js', '.jsx', '.ts', '.tsx'] },
  { id: 'python', label: 'Python', exts: ['.py'] },
  { id: 'go', label: 'Go', exts: ['.go'] },
  { id: 'rust', label: 'Rust', exts: ['.rs'] },
  { id: 'java', label: 'Java', exts: ['.java'] },
  { id: 'c_cpp', label: 'C / C++', exts: ['.c', '.cpp', '.h', '.hpp'] },
  { id: 'csharp', label: 'C#', exts: ['.cs'] },
  { id: 'ruby', label: 'Ruby', exts: ['.rb'] },
  { id: 'dart', label: 'Dart', exts: ['.dart'] },
  { id: 'php', label: 'PHP', exts: ['.php'] },
  { id: 'elixir', label: 'Elixir', exts: ['.ex', '.exs'] },
  { id: 'terraform', label: 'Terraform / HCL', exts: ['.tf'] },
]

export function StatusPage() {
  const [status, setStatus] = useState<AdminStatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<ErrorResponse | null>(null)
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date())
  const [adminKey, setAdminKey] = useState<string | null>(null)

  useEffect(() => {
    if (typeof window !== 'undefined') {
      setAdminKey(localStorage.getItem('admin_api_key'))
    }
  }, [])

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
              {import.meta.env.DEV && (
                <div className="pt-2">
                  <label className="text-xs text-muted-foreground">Dev: Admin API Key (stored in localStorage)</label>
                  <div className="flex items-center gap-2 mt-2">
                    <input
                      type="text"
                      value={adminKey ?? ''}
                      onChange={(e) => setAdminKey(e.target.value)}
                      placeholder="Enter admin API key"
                      className="input input-sm flex-1"
                    />
                    <Button size="sm" onClick={() => {
                      setApiKey(adminKey || null)
                      // reflect storage
                      setAdminKey(adminKey || null)
                    }}>Save</Button>
                    <Button size="sm" variant="outline" onClick={() => {
                      setApiKey(null)
                      setAdminKey(null)
                    }}>Clear</Button>
                  </div>
                </div>
              )}
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
                <span className="text-xs text-muted-foreground font-mono truncate max-w-[200px]">
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
                    status.watcher.watching.map((repo: string) => (
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
                {Object.entries(status.repos).map(([name, path]: [string, string]) => (
                  <RepoRow key={name} name={name} path={path} />
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}

function RepoRow({ name, path }: { name: string; path: string }) {
  const [info, setInfo] = useState<{ respect_gitignore?: boolean; ignore_patterns?: string[] } | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<ErrorResponse | null>(null)
  const [editOpen, setEditOpen] = useState(false)

  useEffect(() => {
    let mounted = true
    const load = async () => {
      try {
        const data = await getRepoInfo(name)
        if (mounted) setInfo({ respect_gitignore: data.respect_gitignore, ignore_patterns: data.ignore_patterns || [] })
      } catch (err) {
        if (mounted) setError(err as ErrorResponse)
      }
    }
    load()
    return () => { mounted = false }
  }, [name])

  const [confirmOpen, setConfirmOpen] = useState(false)

  const toggleConfirmed = async () => {
    setLoading(true)
    setError(null)
    try {
      const newVal = !(info?.respect_gitignore ?? true)
      const data = await setRepoGitignore(name, newVal, info?.ignore_patterns)
      setInfo({ respect_gitignore: data.respect_gitignore, ignore_patterns: data.ignore_patterns || info?.ignore_patterns })
      setConfirmOpen(false)
    } catch (err) {
      setError(err as ErrorResponse)
    } finally {
      setLoading(false)
    }
  }

  const savePatterns = async (patterns: string[] | null) => {
    setLoading(true)
    setError(null)
    try {
      const respect = info?.respect_gitignore ?? true
      const data = await setRepoGitignore(name, respect, patterns ?? undefined)
      setInfo({ respect_gitignore: data.respect_gitignore, ignore_patterns: data.ignore_patterns || [] })
    } catch (err) {
      setError(err as ErrorResponse)
      throw err
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex items-start justify-between border-b pb-2 last:border-0">
      <div className="flex-1">
        <div className="font-medium">{name}</div>
        <div className="text-xs text-muted-foreground font-mono truncate">{path}</div>
        {error && <div className="text-xs text-red-600">Error: {error.detail ?? error.error}</div>}
      </div>
      <div className="flex items-center gap-2">
        <div className="text-sm text-muted-foreground">Respect .gitignore</div>
        <div>
          <Button size="sm" onClick={() => setConfirmOpen(true)} disabled={loading} variant={info?.respect_gitignore ? 'default' : 'outline'}>
            {loading ? 'Updating...' : (info?.respect_gitignore ? 'On' : 'Off')}
          </Button>
        </div>
        <div>
          <Button size="sm" variant="outline" onClick={() => setEditOpen(true)}>Edit Ignore Patterns</Button>
        </div>
      </div>

      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Persist repository setting?</DialogTitle>
            <DialogDescription>
              Changing this will update <span className="font-mono">./config.json</span> in the project and create a timestamped backup. This operation is reversible by restoring the backup.
            </DialogDescription>
          </DialogHeader>
          <div className="mt-4">
            <p className="text-sm">Are you sure you want to turn <strong>{info?.respect_gitignore ? 'off' : 'on'}</strong> <span className="font-mono">respect_gitignore</span> for <strong>{name}</strong>?</p>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmOpen(false)}>Cancel</Button>
            <Button data-testid={`confirm-repo-gitignore-${name}`} onClick={toggleConfirmed} disabled={loading}>
              {loading ? 'Applying...' : 'Confirm and Persist'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <EditPatternsDialog name={name} open={editOpen} onOpenChange={setEditOpen} currentPatterns={info?.ignore_patterns} onSave={savePatterns} />
    </div>
  )
}

function EditPatternsDialog({ name, open, onOpenChange, currentPatterns, onSave }: { name: string; open: boolean; onOpenChange: (v: boolean) => void; currentPatterns?: string[]; onSave: (patterns: string[] | null) => Promise<void> }) {
  const [text, setText] = useState((currentPatterns || []).join('\n'))
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<ErrorResponse | null>(null)

  useEffect(() => {
    setText((currentPatterns || []).join('\n'))
  }, [currentPatterns, open])

  const save = async () => {
    setSaving(true)
    setError(null)
    try {
      const lines = text.split('\n').map((l) => l.trim()).filter((l) => l.length > 0)
      await onSave(lines)
      onOpenChange(false)
    } catch (err) {
      setError(err as ErrorResponse)
    } finally {
      setSaving(false)
    }
  }

  const [selectedLangs, setSelectedLangs] = useState<string[]>([])

  const toggleLang = (id: string) => {
    setSelectedLangs((prev) => (prev.includes(id) ? prev.filter((p) => p !== id) : [...prev, id]))
  }

  const applySelectedLanguages = () => {
    const exts = new Set<string>()
    for (const lang of CODE_LANGUAGES) {
      if (selectedLangs.includes(lang.id)) {
        for (const e of lang.exts) exts.add(e)
      }
    }
    if (exts.size === 0) return
    const lines = Array.from(exts).sort().map((e) => `!**/*${e}`)
    // Merge with existing lines, preferring explicit new lines at top
    const existing = text.split('\n').map((l) => l.trim()).filter((l) => l.length > 0)
    const merged = [...lines, ...existing.filter((l) => !lines.includes(l))]
    setText(merged.join('\n'))
  }

  const selectAll = () => setSelectedLangs(CODE_LANGUAGES.map((l) => l.id))
  const clearAll = () => setSelectedLangs([])

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit ignore patterns for {name}</DialogTitle>
          <DialogDescription>One pattern per line. Prefix with '!' to re-include.</DialogDescription>
        </DialogHeader>
        <div className="mt-4">
          {error && <div className="text-sm text-red-600">Error: {error.detail ?? error.error}</div>}
          <div className="mb-2">
            <div className="flex items-center justify-between mb-2">
              <div className="text-sm font-medium">Preset: Code-only whitelist</div>
              <div className="flex items-center gap-2">
                <Button size="sm" variant="outline" onClick={selectAll}>Select all</Button>
                <Button size="sm" variant="outline" onClick={clearAll}>Clear</Button>
                <Button size="sm" onClick={applySelectedLanguages}>Apply Selected Languages</Button>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 max-h-40 overflow-auto border rounded p-2 bg-white/5">
              {CODE_LANGUAGES.map((lang) => (
                <label key={lang.id} className="flex items-center gap-2">
                  <input type="checkbox" checked={selectedLangs.includes(lang.id)} onChange={() => toggleLang(lang.id)} />
                  <span className="text-sm">{lang.label}</span>
                </label>
              ))}
            </div>
          </div>
          <textarea className="w-full h-48 font-mono text-sm bg-muted p-2" value={text} onChange={(e) => setText(e.target.value)} />
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={saving}>Cancel</Button>
          <Button onClick={save} disabled={saving}>{saving ? 'Saving...' : 'Save Patterns'}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}


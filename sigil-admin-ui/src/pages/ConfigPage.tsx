// Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial licenses are available. Contact: davetmire85@gmail.com

import { useEffect, useState } from 'react'
import { getConfig, saveConfig, getStatus, type ConfigResponse, type ErrorResponse, type AdminStatusResponse } from '@/utils/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { LoadingSpinner } from '@/components/LoadingSpinner'
import { ErrorAlert } from '@/components/ErrorAlert'
import { RefreshCw, Copy } from 'lucide-react'

export function ConfigPage() {
  const [config, setConfig] = useState<ConfigResponse | null>(null)
  const [configText, setConfigText] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<ErrorResponse | null>(null)
  const [status, setStatus] = useState<AdminStatusResponse | null>(null)

  const fetchConfig = async () => {
    try {
      setLoading(true)
      setError(null)
      const [cfg, stat] = await Promise.all([getConfig(), getStatus()])
      setConfig(cfg)
      setConfigText(JSON.stringify(cfg, null, 2))
      setStatus(stat)
    } catch (err) {
      setError(err as ErrorResponse)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchConfig()
  }, [])

  const copyToClipboard = async () => {
    if (config) {
      const text = JSON.stringify(config, null, 2)
      await navigator.clipboard.writeText(text)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Configuration</h1>
          <p className="text-muted-foreground">View current server configuration (read-only)</p>
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={fetchConfig} disabled={loading} variant="outline">
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button onClick={copyToClipboard} variant="outline" disabled={!config}>
            <Copy className="h-4 w-4 mr-2" />
            Copy JSON
          </Button>
          {status?.admin?.mode === 'dev' && (
            <Button onClick={async () => {
              try {
                setSaving(true)
                setError(null)
                const parsed = JSON.parse(configText)
                const resp = await saveConfig(parsed)
                setConfig(resp.config)
                setConfigText(JSON.stringify(resp.config, null, 2))
              } catch (err: any) {
                setError(err as ErrorResponse || { error: 'invalid_json', detail: err?.message })
              } finally {
                setSaving(false)
              }
            }} disabled={!config || saving}>
              {saving ? 'Saving...' : 'Save'}
            </Button>
          )}
        </div>
      </div>

      {error && <ErrorAlert error={error} />}

      {loading && !config ? (
        <LoadingSpinner />
      ) : config ? (
        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
            <CardDescription>
              Full server configuration as JSON {status?.admin?.mode === 'dev' ? '(editable in dev mode)' : '(read-only in prod)'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {status?.admin?.mode === 'dev' ? (
              <div className="bg-muted rounded-md p-4 max-h-[600px] overflow-auto">
                <textarea
                  value={configText}
                  onChange={(e) => setConfigText(e.target.value)}
                  className="w-full h-[520px] font-mono text-sm bg-transparent outline-none"
                  spellCheck={false}
                />
              </div>
            ) : (
              <div className="bg-muted rounded-md p-4 max-h-[600px] overflow-auto">
                <pre className="text-sm font-mono whitespace-pre-wrap break-words">
                  {JSON.stringify(config, null, 2)}
                </pre>
              </div>
            )}
          </CardContent>
        </Card>
      ) : null}
    </div>
  )
}

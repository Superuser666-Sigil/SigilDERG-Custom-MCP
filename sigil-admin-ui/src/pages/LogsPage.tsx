// Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial licenses are available. Contact: davetmire85@gmail.com

import { useEffect, useState, useRef } from 'react'
import { getLogsTail } from '@/utils/api'
import type { LogsResponse, ErrorResponse } from '@/types/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { LoadingSpinner } from '@/components/LoadingSpinner'
import { ErrorAlert } from '@/components/ErrorAlert'
import { RefreshCw, Copy, Play, Pause } from 'lucide-react'
import { Badge } from '@/components/ui/badge'

export function LogsPage() {
  const [logs, setLogs] = useState<LogsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<ErrorResponse | null>(null)
  const [lineCount, setLineCount] = useState(200)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const refreshIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)

  const fetchLogs = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await getLogsTail(lineCount)
      setLogs(data)
      if (typeof window !== 'undefined') {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
      }
    } catch (err) {
      setError(err as ErrorResponse)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchLogs()
  }, [lineCount])

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(fetchLogs, 3000) // Refresh every 3 seconds
      refreshIntervalRef.current = interval
      return () => {
        clearInterval(interval)
        refreshIntervalRef.current = null
      }
    }

    if (refreshIntervalRef.current) {
      clearInterval(refreshIntervalRef.current)
      refreshIntervalRef.current = null
    }
  }, [autoRefresh])

  const copyToClipboard = async () => {
    if (logs?.lines) {
      const text = logs.lines.join('\n')
      await navigator.clipboard.writeText(text)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Server Logs</h1>
          <p className="text-muted-foreground">View recent server log entries</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium">Lines:</label>
            <Input
              type="number"
              value={lineCount}
              onChange={(e) => setLineCount(Math.max(1, Math.min(2000, parseInt(e.target.value, 10) || 200)))}
              className="w-24"
              min={1}
              max={2000}
            />
          </div>
          <Button
            onClick={() => setAutoRefresh(!autoRefresh)}
            variant={autoRefresh ? 'default' : 'outline'}
          >
            {autoRefresh ? (
              <>
                <Pause className="h-4 w-4 mr-2" />
                Pause
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Auto-refresh
              </>
            )}
          </Button>
          <Button onClick={fetchLogs} disabled={loading} variant="outline">
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button onClick={copyToClipboard} variant="outline" disabled={!logs}>
            <Copy className="h-4 w-4 mr-2" />
            Copy
          </Button>
        </div>
      </div>

      {error && <ErrorAlert error={error} />}

      {loading && !logs ? (
        <LoadingSpinner />
      ) : logs ? (
        <Card>
          <CardHeader>
            <CardTitle>Log File</CardTitle>
            <CardDescription className="flex items-center gap-2">
              <span>{logs.path}</span>
              {autoRefresh && <Badge variant="secondary">Auto-refreshing</Badge>}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="bg-muted rounded-md p-4 max-h-[600px] overflow-auto font-mono text-sm">
              {logs.lines.length > 0 ? (
                logs.lines.map((line: string, index: number) => (
                  <div key={index} className="whitespace-pre-wrap break-words">
                    {line}
                  </div>
                ))
              ) : (
                <div className="text-muted-foreground">No log entries</div>
              )}
              <div ref={logsEndRef} />
            </div>
          </CardContent>
        </Card>
      ) : null}
    </div>
  )
}



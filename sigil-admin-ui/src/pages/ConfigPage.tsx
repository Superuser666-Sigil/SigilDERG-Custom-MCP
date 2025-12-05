import { useEffect, useState } from 'react'
import { getConfig, type ConfigResponse, type ErrorResponse } from '@/utils/api'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { LoadingSpinner } from '@/components/LoadingSpinner'
import { ErrorAlert } from '@/components/ErrorAlert'
import { RefreshCw, Copy } from 'lucide-react'

export function ConfigPage() {
  const [config, setConfig] = useState<ConfigResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<ErrorResponse | null>(null)

  const fetchConfig = async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await getConfig()
      setConfig(data)
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
        </div>
      </div>

      {error && <ErrorAlert error={error} />}

      {loading && !config ? (
        <LoadingSpinner />
      ) : config ? (
        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
            <CardDescription>Full server configuration as JSON</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="bg-muted rounded-md p-4 max-h-[600px] overflow-auto">
              <pre className="text-sm font-mono whitespace-pre-wrap break-words">
                {JSON.stringify(config, null, 2)}
              </pre>
            </div>
          </CardContent>
        </Card>
      ) : null}
    </div>
  )
}



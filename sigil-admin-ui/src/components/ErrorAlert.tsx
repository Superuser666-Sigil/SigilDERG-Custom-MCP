// Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial licenses are available. Contact: davetmire85@gmail.com

import { Alert, AlertDescription, AlertTitle } from './ui/alert'
import { AlertCircle } from 'lucide-react'
import { ErrorResponse } from '@/types/api'

interface ErrorAlertProps {
  error: ErrorResponse | Error | string | null
  title?: string
}

export function ErrorAlert({ error, title = 'Error' }: ErrorAlertProps) {
  if (!error) return null

  let message = 'An unexpected error occurred'
  if (typeof error === 'string') {
    message = error
  } else if (error instanceof Error) {
    message = error.message
  } else if ('detail' in error && error.detail) {
    message = error.detail
  } else if ('error' in error) {
    message = error.error
  }

  return (
    <Alert variant="destructive" className="mt-4">
      <AlertCircle className="h-4 w-4" />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>{message}</AlertDescription>
    </Alert>
  )
}


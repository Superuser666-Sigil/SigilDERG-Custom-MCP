import { Badge } from './ui/badge'
import { CheckCircle2, XCircle, AlertCircle } from 'lucide-react'

interface StatusBadgeProps {
  status: boolean | 'enabled' | 'disabled' | 'active' | 'inactive'
  label?: string
}

export function StatusBadge({ status, label }: StatusBadgeProps) {
  const isActive = status === true || status === 'enabled' || status === 'active'

  return (
    <Badge variant={isActive ? 'default' : 'secondary'} className="flex items-center gap-1">
      {isActive ? (
        <CheckCircle2 className="h-3 w-3" />
      ) : (
        <XCircle className="h-3 w-3" />
      )}
      {label || (isActive ? 'Active' : 'Inactive')}
    </Badge>
  )
}



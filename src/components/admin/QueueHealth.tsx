"use client";

import { AlertCircle, AlertTriangle, CheckCircle } from 'lucide-react';

interface QueueHealthProps {
  health: 'healthy' | 'degraded' | 'unhealthy';
}

export default function QueueHealth({ health }: QueueHealthProps) {
  const config = {
    healthy: {
      icon: CheckCircle,
      color: 'text-success',
      bgColor: 'bg-success/10',
      borderColor: 'border-success/30',
      title: 'Queue Healthy',
      description: 'All systems operating normally'
    },
    degraded: {
      icon: AlertTriangle,
      color: 'text-accent',
      bgColor: 'bg-accent-soft',
      borderColor: 'border-accent/30',
      title: 'Queue Degraded',
      description: 'Performance issues detected, monitoring required'
    },
    unhealthy: {
      icon: AlertCircle,
      color: 'text-danger',
      bgColor: 'bg-danger/10',
      borderColor: 'border-danger/30',
      title: 'Queue Unhealthy',
      description: 'Critical issues detected, immediate attention required'
    }
  };

  const { icon: Icon, color, bgColor, borderColor, title, description } = config[health];

  return (
    <div className={`${bgColor} ${borderColor} border rounded p-4`}>
      <div className="flex items-center gap-3">
        <Icon className={`w-6 h-6 ${color}`} />
        <div className="flex-1">
          <h3 className={`font-serif font-semibold ${color}`}>{title}</h3>
          <p className="font-serif text-sm text-subtle">{description}</p>
        </div>
      </div>
    </div>
  );
}

"use client";

import { AlertCircle, AlertTriangle, CheckCircle } from 'lucide-react';

interface QueueHealthProps {
  health: 'healthy' | 'degraded' | 'unhealthy';
}

export default function QueueHealth({ health }: QueueHealthProps) {
  const config = {
    healthy: {
      icon: CheckCircle,
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200',
      title: 'Queue Healthy',
      description: 'All systems operating normally'
    },
    degraded: {
      icon: AlertTriangle,
      color: 'text-yellow-600',
      bgColor: 'bg-yellow-50',
      borderColor: 'border-yellow-200',
      title: 'Queue Degraded',
      description: 'Performance issues detected, monitoring required'
    },
    unhealthy: {
      icon: AlertCircle,
      color: 'text-red-600',
      bgColor: 'bg-red-50',
      borderColor: 'border-red-200',
      title: 'Queue Unhealthy',
      description: 'Critical issues detected, immediate attention required'
    }
  };

  const { icon: Icon, color, bgColor, borderColor, title, description } = config[health];

  return (
    <div className={`${bgColor} ${borderColor} border rounded-lg p-4`}>
      <div className="flex items-center gap-3">
        <Icon className={`w-6 h-6 ${color}`} />
        <div className="flex-1">
          <h3 className={`font-semibold ${color}`}>{title}</h3>
          <p className="text-sm text-gray-600">{description}</p>
        </div>
      </div>
    </div>
  );
}

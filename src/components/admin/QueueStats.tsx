"use client";

import { Activity, CheckCircle, XCircle, Clock, Users } from 'lucide-react';

interface QueueStatsProps {
  stats: {
    queue_depth: number;
    jobs_processing: number;
    jobs_completed_1h: number;
    jobs_failed_1h: number;
    worker_count: number;
    workers_active: number;
    avg_processing_time: number;
    success_rate_24h: number;
  };
}

export default function QueueStats({ stats }: QueueStatsProps) {
  const statCards = [
    {
      label: 'Queue Depth',
      value: stats.queue_depth,
      icon: Activity,
      color: stats.queue_depth > 20 ? 'text-danger' : 'text-accent',
      bgColor: stats.queue_depth > 20 ? 'bg-danger/10' : 'bg-accent-soft'
    },
    {
      label: 'Processing',
      value: stats.jobs_processing,
      icon: Clock,
      color: 'text-subtle',
      bgColor: 'bg-desk'
    },
    {
      label: 'Completed (1h)',
      value: stats.jobs_completed_1h,
      icon: CheckCircle,
      color: 'text-success',
      bgColor: 'bg-success/10'
    },
    {
      label: 'Failed (1h)',
      value: stats.jobs_failed_1h,
      icon: XCircle,
      color: 'text-danger',
      bgColor: 'bg-danger/10'
    },
    {
      label: 'Workers',
      value: `${stats.workers_active}/${stats.worker_count}`,
      icon: Users,
      color: 'text-ink-2',
      bgColor: 'bg-desk'
    },
    {
      label: 'Success Rate (24h)',
      value: `${stats.success_rate_24h.toFixed(1)}%`,
      icon: CheckCircle,
      color: stats.success_rate_24h > 90 ? 'text-success' : 'text-danger',
      bgColor: stats.success_rate_24h > 90 ? 'bg-success/10' : 'bg-danger/10'
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
      {statCards.map((stat) => {
        const Icon = stat.icon;
        return (
          <div
            key={stat.label}
            className="surface-card p-4"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="font-serif text-sm text-subtle">{stat.label}</span>
              <div className={`p-2 rounded-sm ${stat.bgColor}`}>
                <Icon className={`w-4 h-4 ${stat.color}`} />
              </div>
            </div>
            <div className={`font-mono text-2xl font-semibold ${stat.color}`}>
              {stat.value}
            </div>
          </div>
        );
      })}
    </div>
  );
}

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
      color: stats.queue_depth > 20 ? 'text-orange-500' : 'text-blue-500',
      bgColor: stats.queue_depth > 20 ? 'bg-orange-50' : 'bg-blue-50'
    },
    {
      label: 'Processing',
      value: stats.jobs_processing,
      icon: Clock,
      color: 'text-yellow-500',
      bgColor: 'bg-yellow-50'
    },
    {
      label: 'Completed (1h)',
      value: stats.jobs_completed_1h,
      icon: CheckCircle,
      color: 'text-green-500',
      bgColor: 'bg-green-50'
    },
    {
      label: 'Failed (1h)',
      value: stats.jobs_failed_1h,
      icon: XCircle,
      color: 'text-red-500',
      bgColor: 'bg-red-50'
    },
    {
      label: 'Workers',
      value: `${stats.workers_active}/${stats.worker_count}`,
      icon: Users,
      color: 'text-purple-500',
      bgColor: 'bg-purple-50'
    },
    {
      label: 'Success Rate (24h)',
      value: `${stats.success_rate_24h.toFixed(1)}%`,
      icon: CheckCircle,
      color: stats.success_rate_24h > 90 ? 'text-green-500' : 'text-orange-500',
      bgColor: stats.success_rate_24h > 90 ? 'bg-green-50' : 'bg-orange-50'
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
      {statCards.map((stat) => {
        const Icon = stat.icon;
        return (
          <div
            key={stat.label}
            className="bg-white border rounded-lg p-4 hover:shadow-md transition-shadow"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-500">{stat.label}</span>
              <div className={`p-2 rounded-lg ${stat.bgColor}`}>
                <Icon className={`w-4 h-4 ${stat.color}`} />
              </div>
            </div>
            <div className={`text-2xl font-bold ${stat.color}`}>
              {stat.value}
            </div>
          </div>
        );
      })}
    </div>
  );
}

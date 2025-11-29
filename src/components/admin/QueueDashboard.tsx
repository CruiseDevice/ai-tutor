"use client";

import { useEffect, useState, useCallback } from 'react';
import { adminApi } from '@/lib/api-client';
import QueueStats from './QueueStats';
import JobsList from './JobsList';
import QueueHealth from './QueueHealth';
import { ToastContainer } from './Toast';
import ConfirmDialog from './ConfirmDialog';
import { useToast } from '@/hooks/useToast';
import { RefreshCw, AlertCircle } from 'lucide-react';

interface QueueStatsData {
  queue_depth: number;
  jobs_processing: number;
  jobs_pending: number;
  jobs_completed_1h: number;
  jobs_failed_1h: number;
  worker_count: number;
  workers_active: number;
  avg_processing_time: number;
  success_rate_24h: number;
  queue_health: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
}

interface Job {
  job_id: string;
  status: string;
  document_id?: string;
  enqueue_time?: string;
  start_time?: string;
  finish_time?: string;
  error?: string;
}

export default function QueueDashboard() {
  const [stats, setStats] = useState<QueueStatsData | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [useSSE, setUseSSE] = useState(true);
  const [statusFilter, setStatusFilter] = useState<string | undefined>(undefined);
  const [confirmDialog, setConfirmDialog] = useState<{
    isOpen: boolean;
    title: string;
    message: string;
    onConfirm: () => void;
  } | null>(null);

  const toast = useToast();

  // Fetch initial data
  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch stats and jobs in parallel
      const [statsRes, jobsRes] = await Promise.all([
        adminApi.getQueueStats(),
        adminApi.listJobs(statusFilter, 50)
      ]);

      if (!statsRes.ok || !jobsRes.ok) {
        throw new Error('Failed to fetch queue data');
      }

      const [statsData, jobsData] = await Promise.all([
        statsRes.json(),
        jobsRes.json()
      ]);

      setStats(statsData);
      setJobs(jobsData.jobs);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Failed to fetch queue data:', err);
    } finally {
      setLoading(false);
    }
  }, [statusFilter]);

  // SSE connection for real-time updates
  useEffect(() => {
    if (!useSSE) return;

    const eventSource = adminApi.createQueueStatsStream();

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setStats(data);
    };

    eventSource.onerror = (err) => {
      console.error('SSE error:', err);
      eventSource.close();
      // Fall back to polling
      setUseSSE(false);
    };

    return () => eventSource.close();
  }, [useSSE]);

  // Polling fallback (if SSE fails)
  useEffect(() => {
    if (useSSE) return;

    const interval = setInterval(fetchData, 5000); // Poll every 5s
    return () => clearInterval(interval);
  }, [useSSE, fetchData]);

  // Initial fetch
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Retry job handler
  const handleRetryJob = async (documentId: string) => {
    try {
      const res = await adminApi.retryJob(documentId);
      if (!res.ok) throw new Error('Failed to retry job');

      toast.success('Job retried successfully');
      // Refresh data
      await fetchData();
    } catch (err) {
      toast.error('Failed to retry job: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  // Cancel job handler
  const handleCancelJob = async (jobId: string) => {
    setConfirmDialog({
      isOpen: true,
      title: 'Cancel Job',
      message: 'Are you sure you want to cancel this job? This action cannot be undone.',
      onConfirm: async () => {
        try {
          const res = await adminApi.cancelJob(jobId);
          if (!res.ok) throw new Error('Failed to cancel job');

          toast.success('Job cancelled successfully');
          // Refresh data
          await fetchData();
        } catch (err) {
          toast.error('Failed to cancel job: ' + (err instanceof Error ? err.message : 'Unknown error'));
        } finally {
          setConfirmDialog(null);
        }
      },
    });
  };

  // Retry all failed jobs
  const handleRetryAllFailed = () => {
    const failedJobs = jobs.filter(job => job.status === 'failed' && job.document_id);

    if (failedJobs.length === 0) {
      toast.info('No failed jobs to retry');
      return;
    }

    setConfirmDialog({
      isOpen: true,
      title: 'Retry All Failed Jobs',
      message: `Are you sure you want to retry ${failedJobs.length} failed job${failedJobs.length > 1 ? 's' : ''}?`,
      onConfirm: async () => {
        let successCount = 0;
        let failCount = 0;

        for (const job of failedJobs) {
          try {
            const res = await adminApi.retryJob(job.document_id!);
            if (res.ok) {
              successCount++;
            } else {
              failCount++;
            }
          } catch {
            failCount++;
          }
        }

        if (successCount > 0) {
          toast.success(`Successfully retried ${successCount} job${successCount > 1 ? 's' : ''}`);
        }
        if (failCount > 0) {
          toast.error(`Failed to retry ${failCount} job${failCount > 1 ? 's' : ''}`);
        }

        await fetchData();
        setConfirmDialog(null);
      },
    });
  };

  // Cancel all pending jobs
  const handleCancelAllPending = () => {
    const pendingJobs = jobs.filter(job => job.status === 'queued');

    if (pendingJobs.length === 0) {
      toast.info('No pending jobs to cancel');
      return;
    }

    setConfirmDialog({
      isOpen: true,
      title: 'Cancel All Pending Jobs',
      message: `Are you sure you want to cancel ${pendingJobs.length} pending job${pendingJobs.length > 1 ? 's' : ''}? This action cannot be undone.`,
      onConfirm: async () => {
        let successCount = 0;
        let failCount = 0;

        for (const job of pendingJobs) {
          try {
            const res = await adminApi.cancelJob(job.job_id);
            if (res.ok) {
              successCount++;
            } else {
              failCount++;
            }
          } catch {
            failCount++;
          }
        }

        if (successCount > 0) {
          toast.success(`Successfully cancelled ${successCount} job${successCount > 1 ? 's' : ''}`);
        }
        if (failCount > 0) {
          toast.error(`Failed to cancel ${failCount} job${failCount > 1 ? 's' : ''}`);
        }

        await fetchData();
        setConfirmDialog(null);
      },
    });
  };

  if (loading && !stats) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 animate-spin text-gray-400" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex items-center gap-2 text-red-800">
          <AlertCircle className="w-5 h-5" />
          <span className="font-semibold">Error loading queue data</span>
        </div>
        <p className="text-red-600 text-sm mt-1">{error}</p>
        <button
          onClick={fetchData}
          className="mt-3 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Queue Monitor</h1>
          <p className="text-gray-500 text-sm mt-1">
            Real-time document processing queue status
          </p>
        </div>

        <div className="flex gap-3">
          {/* Bulk Actions */}
          <div className="flex gap-2">
            <button
              onClick={handleRetryAllFailed}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={jobs.filter(j => j.status === 'failed').length === 0}
            >
              <RefreshCw className="w-4 h-4" />
              Retry All Failed
            </button>
            <button
              onClick={handleCancelAllPending}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={jobs.filter(j => j.status === 'queued').length === 0}
            >
              Cancel All Pending
            </button>
          </div>

          <button
            onClick={fetchData}
            className="flex items-center gap-2 px-4 py-2 border rounded-lg hover:bg-gray-50"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </div>

      {/* Queue Health */}
      {stats && <QueueHealth health={stats.queue_health} />}

      {/* Stats Cards */}
      {stats && <QueueStats stats={stats} />}

      {/* Jobs List */}
      <JobsList
        jobs={jobs}
        statusFilter={statusFilter}
        onFilterChange={setStatusFilter}
        onRetryJob={handleRetryJob}
        onCancelJob={handleCancelJob}
      />

      {/* Live indicator */}
      <div className="text-xs text-gray-400 flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${useSSE ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
        {useSSE ? 'Live updates enabled' : 'Polling mode (5s refresh)'}
      </div>

      {/* Toast Notifications */}
      <ToastContainer toasts={toast.toasts} onClose={toast.closeToast} />

      {/* Confirmation Dialog */}
      {confirmDialog && (
        <ConfirmDialog
          isOpen={confirmDialog.isOpen}
          title={confirmDialog.title}
          message={confirmDialog.message}
          variant="danger"
          confirmText="Cancel Job"
          onConfirm={confirmDialog.onConfirm}
          onCancel={() => setConfirmDialog(null)}
        />
      )}
    </div>
  );
}

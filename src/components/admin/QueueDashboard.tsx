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
        <RefreshCw className="w-8 h-8 animate-spin text-faint" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-danger/10 border border-danger/30 rounded p-4">
        <div className="flex items-center gap-2 text-danger">
          <AlertCircle className="w-5 h-5" />
          <span className="font-serif font-semibold">Error loading queue data</span>
        </div>
        <p className="font-serif text-danger text-sm mt-1">{error}</p>
        <button
          onClick={fetchData}
          className="btn mt-3 text-sm"
          style={{ background: 'var(--danger)', borderColor: 'var(--danger)', color: '#fff' }}
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="font-serif text-3xl font-semibold text-ink">Queue Monitor</h1>
          <p className="font-serif text-subtle text-sm mt-1">
            Real-time document processing queue status
          </p>
        </div>

        <div className="flex gap-3 flex-wrap">
          {/* Bulk Actions */}
          <div className="flex gap-2">
            <button
              onClick={handleRetryAllFailed}
              className="btn btn-primary text-sm disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={jobs.filter(j => j.status === 'failed').length === 0}
            >
              <RefreshCw className="w-4 h-4" />
              Retry All Failed
            </button>
            <button
              onClick={handleCancelAllPending}
              className="btn text-sm disabled:opacity-50 disabled:cursor-not-allowed"
              style={{ background: 'var(--danger)', borderColor: 'var(--danger)', color: '#fff' }}
              disabled={jobs.filter(j => j.status === 'queued').length === 0}
            >
              Cancel All Pending
            </button>
          </div>

          <button
            onClick={fetchData}
            className="btn btn-quiet text-sm"
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
      <div className="font-mono text-xs text-faint flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${useSSE ? 'bg-success animate-pulse' : 'bg-faint'}`} />
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

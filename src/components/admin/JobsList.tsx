"use client";

import { useState } from 'react';
import { RefreshCw, X, Clock, CheckCircle, XCircle, Loader2, Eye } from 'lucide-react';
import JobDetails from './JobDetails';

interface Job {
  job_id: string;
  status: string;
  document_id?: string;
  enqueue_time?: string;
  start_time?: string;
  finish_time?: string;
  error?: string;
  function?: string;
}

interface JobsListProps {
  jobs: Job[];
  statusFilter?: string;
  onFilterChange: (filter: string | undefined) => void;
  onRetryJob: (documentId: string) => Promise<void>;
  onCancelJob: (jobId: string) => Promise<void>;
}

export default function JobsList({
  jobs,
  statusFilter,
  onFilterChange,
  onRetryJob,
  onCancelJob
}: JobsListProps) {
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [loadingAction, setLoadingAction] = useState<string | null>(null);

  const handleRetry = async (documentId: string, jobId: string) => {
    setLoadingAction(`retry-${jobId}`);
    try {
      await onRetryJob(documentId);
    } finally {
      setLoadingAction(null);
    }
  };

  const handleCancel = async (jobId: string) => {
    setLoadingAction(`cancel-${jobId}`);
    try {
      await onCancelJob(jobId);
    } finally {
      setLoadingAction(null);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'complete':
        return <CheckCircle className="w-4 h-4 text-success" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-danger" />;
      case 'in_progress':
        return <Loader2 className="w-4 h-4 text-accent animate-spin" />;
      case 'queued':
        return <Clock className="w-4 h-4 text-subtle" />;
      default:
        return <Clock className="w-4 h-4 text-faint" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const colors = {
      complete: 'bg-success/15 text-success',
      failed: 'bg-danger/15 text-danger',
      in_progress: 'bg-accent/15 text-accent',
      queued: 'bg-desk text-subtle'
    };

    return (
      <span className={`px-2 py-1 rounded-sm text-xs font-medium ${colors[status as keyof typeof colors] || 'bg-desk text-subtle'}`}>
        {status}
      </span>
    );
  };

  return (
    <div className="surface-card overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-hair">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <h2 className="font-serif text-lg font-semibold text-ink">Jobs</h2>

          {/* Filter pills */}
          <div className="flex gap-1">
            <button
              onClick={() => onFilterChange(undefined)}
              className={`px-3 py-1 font-serif text-sm rounded-sm transition-colors ${!statusFilter ? 'bg-accent text-paper' : 'text-subtle hover:text-ink hover:bg-desk'}`}
            >
              All
            </button>
            <button
              onClick={() => onFilterChange('queued')}
              className={`px-3 py-1 font-serif text-sm rounded-sm transition-colors ${statusFilter === 'queued' ? 'bg-accent text-paper' : 'text-subtle hover:text-ink hover:bg-desk'}`}
            >
              Queued
            </button>
            <button
              onClick={() => onFilterChange('in_progress')}
              className={`px-3 py-1 font-serif text-sm rounded-sm transition-colors ${statusFilter === 'in_progress' ? 'bg-accent text-paper' : 'text-subtle hover:text-ink hover:bg-desk'}`}
            >
              Processing
            </button>
            <button
              onClick={() => onFilterChange('failed')}
              className={`px-3 py-1 font-serif text-sm rounded-sm transition-colors ${statusFilter === 'failed' ? 'bg-accent text-paper' : 'text-subtle hover:text-ink hover:bg-desk'}`}
            >
              Failed
            </button>
          </div>
        </div>
      </div>

      {/* Jobs table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-desk border-b border-hair">
            <tr>
              <th className="px-4 py-3 text-left font-mono text-xs font-normal text-faint uppercase tracking-wider">Status</th>
              <th className="px-4 py-3 text-left font-mono text-xs font-normal text-faint uppercase tracking-wider">Job ID</th>
              <th className="px-4 py-3 text-left font-mono text-xs font-normal text-faint uppercase tracking-wider">Document ID</th>
              <th className="px-4 py-3 text-left font-mono text-xs font-normal text-faint uppercase tracking-wider">Enqueued</th>
              <th className="px-4 py-3 text-left font-mono text-xs font-normal text-faint uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-hair">
            {jobs.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center font-serif text-subtle">
                  No jobs found
                </td>
              </tr>
            ) : (
              jobs.map((job) => (
                <tr key={job.job_id} className="hover:bg-desk/50 transition-colors">
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(job.status)}
                      {getStatusBadge(job.status)}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <code className="text-xs text-ink-2">
                      {job.job_id.substring(0, 8)}...
                    </code>
                  </td>
                  <td className="px-4 py-3">
                    <code className="text-xs text-ink-2">
                      {job.document_id ? job.document_id.substring(0, 8) + '...' : 'N/A'}
                    </code>
                  </td>
                  <td className="px-4 py-3 font-serif text-sm text-subtle">
                    {job.enqueue_time ? new Date(job.enqueue_time).toLocaleString() : 'N/A'}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2">
                      <button
                        onClick={() => setSelectedJob(job)}
                        className="p-1 hover:bg-accent-soft rounded-sm transition-colors"
                        title="View details"
                      >
                        <Eye className="w-4 h-4 text-subtle" />
                      </button>
                      {job.status === 'failed' && job.document_id && (
                        <button
                          onClick={() => handleRetry(job.document_id!, job.job_id)}
                          disabled={loadingAction === `retry-${job.job_id}`}
                          className="p-1 hover:bg-accent-soft rounded-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                          title="Retry job"
                        >
                          <RefreshCw className={`w-4 h-4 text-accent ${loadingAction === `retry-${job.job_id}` ? 'animate-spin' : ''}`} />
                        </button>
                      )}
                      {(job.status === 'queued' || job.status === 'in_progress') && (
                        <button
                          onClick={() => handleCancel(job.job_id)}
                          disabled={loadingAction === `cancel-${job.job_id}`}
                          className="p-1 hover:bg-danger/10 rounded-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                          title="Cancel job"
                        >
                          {loadingAction === `cancel-${job.job_id}` ? (
                            <Loader2 className="w-4 h-4 text-danger animate-spin" />
                          ) : (
                            <X className="w-4 h-4 text-danger" />
                          )}
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Job Details Modal */}
      {selectedJob && (
        <JobDetails
          job={selectedJob}
          onClose={() => setSelectedJob(null)}
          onRetry={async (documentId) => {
            await handleRetry(documentId, selectedJob.job_id);
            setSelectedJob(null);
          }}
          onCancel={async (jobId) => {
            await handleCancel(jobId);
            setSelectedJob(null);
          }}
        />
      )}
    </div>
  );
}

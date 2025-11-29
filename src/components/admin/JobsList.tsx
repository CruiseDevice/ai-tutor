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
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />;
      case 'in_progress':
        return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'queued':
        return <Clock className="w-4 h-4 text-yellow-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const colors = {
      complete: 'bg-green-100 text-green-800',
      failed: 'bg-red-100 text-red-800',
      in_progress: 'bg-blue-100 text-blue-800',
      queued: 'bg-yellow-100 text-yellow-800'
    };

    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${colors[status as keyof typeof colors] || 'bg-gray-100 text-gray-800'}`}>
        {status}
      </span>
    );
  };

  return (
    <div className="bg-white border rounded-lg">
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Jobs</h2>

          {/* Filter buttons */}
          <div className="flex gap-2">
            <button
              onClick={() => onFilterChange(undefined)}
              className={`px-3 py-1 text-sm rounded ${!statusFilter ? 'bg-gray-900 text-white' : 'bg-gray-100 hover:bg-gray-200'}`}
            >
              All
            </button>
            <button
              onClick={() => onFilterChange('queued')}
              className={`px-3 py-1 text-sm rounded ${statusFilter === 'queued' ? 'bg-gray-900 text-white' : 'bg-gray-100 hover:bg-gray-200'}`}
            >
              Queued
            </button>
            <button
              onClick={() => onFilterChange('in_progress')}
              className={`px-3 py-1 text-sm rounded ${statusFilter === 'in_progress' ? 'bg-gray-900 text-white' : 'bg-gray-100 hover:bg-gray-200'}`}
            >
              Processing
            </button>
            <button
              onClick={() => onFilterChange('failed')}
              className={`px-3 py-1 text-sm rounded ${statusFilter === 'failed' ? 'bg-gray-900 text-white' : 'bg-gray-100 hover:bg-gray-200'}`}
            >
              Failed
            </button>
          </div>
        </div>
      </div>

      {/* Jobs table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50 border-b">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Job ID</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Document ID</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Enqueued</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {jobs.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-4 py-8 text-center text-gray-500">
                  No jobs found
                </td>
              </tr>
            ) : (
              jobs.map((job) => (
                <tr key={job.job_id} className="hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(job.status)}
                      {getStatusBadge(job.status)}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <code className="text-xs text-gray-600">
                      {job.job_id.substring(0, 8)}...
                    </code>
                  </td>
                  <td className="px-4 py-3">
                    <code className="text-xs text-gray-600">
                      {job.document_id ? job.document_id.substring(0, 8) + '...' : 'N/A'}
                    </code>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-500">
                    {job.enqueue_time ? new Date(job.enqueue_time).toLocaleString() : 'N/A'}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2">
                      <button
                        onClick={() => setSelectedJob(job)}
                        className="p-1 hover:bg-gray-100 rounded"
                        title="View details"
                      >
                        <Eye className="w-4 h-4 text-gray-500" />
                      </button>
                      {job.status === 'failed' && job.document_id && (
                        <button
                          onClick={() => handleRetry(job.document_id!, job.job_id)}
                          disabled={loadingAction === `retry-${job.job_id}`}
                          className="p-1 hover:bg-gray-100 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                          title="Retry job"
                        >
                          <RefreshCw className={`w-4 h-4 text-blue-500 ${loadingAction === `retry-${job.job_id}` ? 'animate-spin' : ''}`} />
                        </button>
                      )}
                      {(job.status === 'queued' || job.status === 'in_progress') && (
                        <button
                          onClick={() => handleCancel(job.job_id)}
                          disabled={loadingAction === `cancel-${job.job_id}`}
                          className="p-1 hover:bg-gray-100 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                          title="Cancel job"
                        >
                          {loadingAction === `cancel-${job.job_id}` ? (
                            <Loader2 className="w-4 h-4 text-red-500 animate-spin" />
                          ) : (
                            <X className="w-4 h-4 text-red-500" />
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

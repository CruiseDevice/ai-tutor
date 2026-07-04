"use client";

import { X, Clock, CheckCircle, XCircle, AlertCircle, Copy } from 'lucide-react';
import { useState } from 'react';

interface JobDetailsProps {
  job: {
    job_id: string;
    status: string;
    document_id?: string;
    enqueue_time?: string;
    start_time?: string;
    finish_time?: string;
    error?: string;
    function?: string;
    args?: unknown[];
    result?: unknown;
  };
  onClose: () => void;
  onRetry?: (documentId: string) => Promise<void>;
  onCancel?: (jobId: string) => Promise<void>;
}

export default function JobDetails({ job, onClose, onRetry, onCancel }: JobDetailsProps) {
  const [copying, setCopying] = useState<string | null>(null);

  const getStatusIcon = () => {
    switch (job.status) {
      case 'complete':
        return <CheckCircle className="w-6 h-6 text-success" />;
      case 'failed':
        return <XCircle className="w-6 h-6 text-danger" />;
      case 'in_progress':
        return <Clock className="w-6 h-6 text-accent animate-spin" />;
      case 'queued':
        return <Clock className="w-6 h-6 text-subtle" />;
      default:
        return <AlertCircle className="w-6 h-6 text-faint" />;
    }
  };

  const getStatusColor = () => {
    switch (job.status) {
      case 'complete':
        return 'bg-success/15 text-success border-success/30';
      case 'failed':
        return 'bg-danger/15 text-danger border-danger/30';
      case 'in_progress':
        return 'bg-accent/15 text-accent border-accent/30';
      case 'queued':
        return 'bg-desk text-subtle border-hair';
      default:
        return 'bg-desk text-subtle border-hair';
    }
  };

  const copyToClipboard = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopying(label);
      setTimeout(() => setCopying(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const formatTimestamp = (timestamp?: string) => {
    if (!timestamp) return 'N/A';
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const calculateDuration = () => {
    if (!job.start_time || !job.finish_time) return null;
    const start = new Date(job.start_time).getTime();
    const finish = new Date(job.finish_time).getTime();
    const durationMs = finish - start;
    const seconds = Math.floor(durationMs / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;

    if (minutes > 0) {
      return `${minutes}m ${remainingSeconds}s`;
    }
    return `${seconds}s`;
  };

  return (
    <div className="fixed inset-0 bg-ink/40 flex items-center justify-center z-50 p-4">
      <div className="bg-paper rounded shadow-page max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-hair">
          <div className="flex items-center gap-3">
            {getStatusIcon()}
            <div>
              <h2 className="font-serif text-xl font-semibold text-ink">Job Details</h2>
              <p className="font-mono text-sm text-faint">ID: {job.job_id.substring(0, 16)}...</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-desk rounded-sm transition-colors"
          >
            <X className="w-5 h-5 text-subtle" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Status Badge */}
          <div className="flex items-center gap-4">
            <span className="font-serif text-sm font-medium text-subtle">Status:</span>
            <span className={`px-3 py-1 rounded-sm font-serif text-sm font-medium border ${getStatusColor()}`}>
              {job.status}
            </span>
          </div>

          {/* Timestamps */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-desk rounded p-4">
              <div className="font-mono text-xs text-faint mb-1">Enqueued</div>
              <div className="font-serif font-medium text-ink">{formatTimestamp(job.enqueue_time)}</div>
            </div>

            <div className="bg-desk rounded p-4">
              <div className="font-mono text-xs text-faint mb-1">Started</div>
              <div className="font-serif font-medium text-ink">{formatTimestamp(job.start_time)}</div>
            </div>

            <div className="bg-desk rounded p-4">
              <div className="font-mono text-xs text-faint mb-1">Finished</div>
              <div className="font-serif font-medium text-ink">{formatTimestamp(job.finish_time)}</div>
            </div>

            {calculateDuration() && (
              <div className="bg-desk rounded p-4">
                <div className="font-mono text-xs text-faint mb-1">Duration</div>
                <div className="font-serif font-medium text-ink">{calculateDuration()}</div>
              </div>
            )}
          </div>

          {/* IDs */}
          <div className="space-y-3">
            <div className="flex items-center justify-between bg-desk rounded p-4">
              <div>
                <div className="font-mono text-xs text-faint mb-1">Job ID</div>
                <code className="text-sm font-mono text-ink-2">{job.job_id}</code>
              </div>
              <button
                onClick={() => copyToClipboard(job.job_id, 'job_id')}
                className="p-2 hover:bg-hair rounded-sm transition-colors"
                title="Copy to clipboard"
              >
                {copying === 'job_id' ? (
                  <CheckCircle className="w-4 h-4 text-success" />
                ) : (
                  <Copy className="w-4 h-4 text-subtle" />
                )}
              </button>
            </div>

            {job.document_id && (
              <div className="flex items-center justify-between bg-desk rounded p-4">
                <div>
                  <div className="font-mono text-xs text-faint mb-1">Document ID</div>
                  <code className="text-sm font-mono text-ink-2">{job.document_id}</code>
                </div>
                <button
                  onClick={() => copyToClipboard(job.document_id!, 'document_id')}
                  className="p-2 hover:bg-hair rounded-sm transition-colors"
                  title="Copy to clipboard"
                >
                  {copying === 'document_id' ? (
                    <CheckCircle className="w-4 h-4 text-success" />
                  ) : (
                    <Copy className="w-4 h-4 text-subtle" />
                  )}
                </button>
              </div>
            )}
          </div>

          {/* Function */}
          {job.function && (
            <div className="bg-desk rounded p-4">
              <div className="font-mono text-xs text-faint mb-1">Function</div>
              <code className="text-sm font-mono text-ink-2">{job.function}</code>
            </div>
          )}

          {/* Error Message */}
          {job.error && (
            <div className="bg-danger/10 border border-danger/30 rounded p-4">
              <div className="flex items-center gap-2 mb-2">
                <XCircle className="w-5 h-5 text-danger" />
                <div className="font-serif font-semibold text-danger">Error</div>
              </div>
              <pre className="text-sm text-danger whitespace-pre-wrap font-mono">
                {job.error}
              </pre>
            </div>
          )}

          {/* Result */}
          {job.result && job.status === 'complete' && (
            <div className="bg-success/10 border border-success/30 rounded p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-5 h-5 text-success" />
                <div className="font-serif font-semibold text-success">Result</div>
              </div>
              <pre className="text-sm text-success whitespace-pre-wrap font-mono max-h-40 overflow-y-auto">
                {typeof job.result === 'string' ? job.result : JSON.stringify(job.result, null, 2)}
              </pre>
            </div>
          )}

          {/* Arguments */}
          {job.args && job.args.length > 0 && (
            <div className="bg-desk rounded p-4">
              <div className="font-mono text-xs text-faint mb-2">Arguments</div>
              <pre className="text-sm font-mono text-ink-2 whitespace-pre-wrap">
                {JSON.stringify(job.args, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="border-t border-hair p-6 bg-desk flex items-center justify-between gap-4 flex-wrap">
          <div className="font-serif text-sm text-subtle">
            {job.status === 'failed' && 'This job failed and can be retried'}
            {job.status === 'queued' && 'This job is waiting to be processed'}
            {job.status === 'in_progress' && 'This job is currently being processed'}
            {job.status === 'complete' && 'This job completed successfully'}
          </div>

          <div className="flex gap-3">
            {job.status === 'failed' && job.document_id && onRetry && (
              <button
                onClick={() => onRetry(job.document_id!)}
                className="btn btn-primary text-sm"
              >
                Retry Job
              </button>
            )}

            {(job.status === 'queued' || job.status === 'in_progress') && onCancel && (
              <button
                onClick={() => onCancel(job.job_id)}
                className="btn text-sm"
                style={{ background: 'var(--danger)', borderColor: 'var(--danger)', color: '#fff' }}
              >
                Cancel Job
              </button>
            )}

            <button
              onClick={onClose}
              className="btn btn-quiet text-sm"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

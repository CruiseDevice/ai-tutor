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
        return <CheckCircle className="w-6 h-6 text-green-500" />;
      case 'failed':
        return <XCircle className="w-6 h-6 text-red-500" />;
      case 'in_progress':
        return <Clock className="w-6 h-6 text-blue-500 animate-spin" />;
      case 'queued':
        return <Clock className="w-6 h-6 text-yellow-500" />;
      default:
        return <AlertCircle className="w-6 h-6 text-gray-400" />;
    }
  };

  const getStatusColor = () => {
    switch (job.status) {
      case 'complete':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'failed':
        return 'bg-red-100 text-red-800 border-red-200';
      case 'in_progress':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'queued':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
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
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div className="flex items-center gap-3">
            {getStatusIcon()}
            <div>
              <h2 className="text-xl font-bold">Job Details</h2>
              <p className="text-sm text-gray-500">ID: {job.job_id.substring(0, 16)}...</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Status Badge */}
          <div className="flex items-center gap-4">
            <span className="text-sm font-medium text-gray-500">Status:</span>
            <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getStatusColor()}`}>
              {job.status}
            </span>
          </div>

          {/* Timestamps */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-xs text-gray-500 mb-1">Enqueued</div>
              <div className="font-medium">{formatTimestamp(job.enqueue_time)}</div>
            </div>

            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-xs text-gray-500 mb-1">Started</div>
              <div className="font-medium">{formatTimestamp(job.start_time)}</div>
            </div>

            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-xs text-gray-500 mb-1">Finished</div>
              <div className="font-medium">{formatTimestamp(job.finish_time)}</div>
            </div>

            {calculateDuration() && (
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="text-xs text-gray-500 mb-1">Duration</div>
                <div className="font-medium">{calculateDuration()}</div>
              </div>
            )}
          </div>

          {/* IDs */}
          <div className="space-y-3">
            <div className="flex items-center justify-between bg-gray-50 rounded-lg p-4">
              <div>
                <div className="text-xs text-gray-500 mb-1">Job ID</div>
                <code className="text-sm font-mono">{job.job_id}</code>
              </div>
              <button
                onClick={() => copyToClipboard(job.job_id, 'job_id')}
                className="p-2 hover:bg-gray-200 rounded transition-colors"
                title="Copy to clipboard"
              >
                {copying === 'job_id' ? (
                  <CheckCircle className="w-4 h-4 text-green-500" />
                ) : (
                  <Copy className="w-4 h-4 text-gray-500" />
                )}
              </button>
            </div>

            {job.document_id && (
              <div className="flex items-center justify-between bg-gray-50 rounded-lg p-4">
                <div>
                  <div className="text-xs text-gray-500 mb-1">Document ID</div>
                  <code className="text-sm font-mono">{job.document_id}</code>
                </div>
                <button
                  onClick={() => copyToClipboard(job.document_id!, 'document_id')}
                  className="p-2 hover:bg-gray-200 rounded transition-colors"
                  title="Copy to clipboard"
                >
                  {copying === 'document_id' ? (
                    <CheckCircle className="w-4 h-4 text-green-500" />
                  ) : (
                    <Copy className="w-4 h-4 text-gray-500" />
                  )}
                </button>
              </div>
            )}
          </div>

          {/* Function */}
          {job.function && (
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-xs text-gray-500 mb-1">Function</div>
              <code className="text-sm font-mono">{job.function}</code>
            </div>
          )}

          {/* Error Message */}
          {job.error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <XCircle className="w-5 h-5 text-red-500" />
                <div className="text-sm font-semibold text-red-800">Error</div>
              </div>
              <pre className="text-sm text-red-700 whitespace-pre-wrap font-mono">
                {job.error}
              </pre>
            </div>
          )}

          {/* Result */}
          {job.result && job.status === 'complete' && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-5 h-5 text-green-500" />
                <div className="text-sm font-semibold text-green-800">Result</div>
              </div>
              <pre className="text-sm text-green-700 whitespace-pre-wrap font-mono max-h-40 overflow-y-auto">
                {typeof job.result === 'string' ? job.result : JSON.stringify(job.result, null, 2)}
              </pre>
            </div>
          )}

          {/* Arguments */}
          {job.args && job.args.length > 0 && (
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-xs text-gray-500 mb-2">Arguments</div>
              <pre className="text-sm font-mono whitespace-pre-wrap">
                {JSON.stringify(job.args, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="border-t p-6 bg-gray-50 flex items-center justify-between">
          <div className="text-sm text-gray-500">
            {job.status === 'failed' && 'This job failed and can be retried'}
            {job.status === 'queued' && 'This job is waiting to be processed'}
            {job.status === 'in_progress' && 'This job is currently being processed'}
            {job.status === 'complete' && 'This job completed successfully'}
          </div>

          <div className="flex gap-3">
            {job.status === 'failed' && job.document_id && onRetry && (
              <button
                onClick={() => onRetry(job.document_id!)}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Retry Job
              </button>
            )}

            {(job.status === 'queued' || job.status === 'in_progress') && onCancel && (
              <button
                onClick={() => onCancel(job.job_id)}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                Cancel Job
              </button>
            )}

            <button
              onClick={onClose}
              className="px-4 py-2 border rounded-lg hover:bg-gray-100 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

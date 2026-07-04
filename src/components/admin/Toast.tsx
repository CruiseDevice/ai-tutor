"use client";

import { useEffect } from 'react';
import { CheckCircle, XCircle, AlertCircle, Info, X } from 'lucide-react';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface ToastMessage {
  id: string;
  type: ToastType;
  message: string;
  duration?: number;
}

interface ToastProps {
  toast: ToastMessage;
  onClose: (id: string) => void;
}

export function Toast({ toast, onClose }: ToastProps) {
  useEffect(() => {
    const duration = toast.duration || 5000;
    const timer = setTimeout(() => {
      onClose(toast.id);
    }, duration);

    return () => clearTimeout(timer);
  }, [toast.id, toast.duration, onClose]);

  const getIcon = () => {
    switch (toast.type) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-success" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-danger" />;
      case 'warning':
        return <AlertCircle className="w-5 h-5 text-accent" />;
      case 'info':
        return <Info className="w-5 h-5 text-accent" />;
    }
  };

  const getStyles = () => {
    switch (toast.type) {
      case 'success':
        return 'bg-success/10 border-success/30 text-success';
      case 'error':
        return 'bg-danger/10 border-danger/30 text-danger';
      case 'warning':
        return 'bg-accent-soft border-accent/30 text-accent-ink';
      case 'info':
        return 'bg-accent-soft border-accent/30 text-accent-ink';
    }
  };

  return (
    <div
      className={`flex items-center gap-3 p-4 rounded border shadow-card min-w-[300px] max-w-md animate-slide-in bg-paper ${getStyles()}`}
    >
      {getIcon()}
      <p className="flex-1 font-serif text-sm font-medium">{toast.message}</p>
      <button
        onClick={() => onClose(toast.id)}
        className="p-1 hover:bg-ink/10 rounded-sm transition-colors"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}

interface ToastContainerProps {
  toasts: ToastMessage[];
  onClose: (id: string) => void;
}

export function ToastContainer({ toasts, onClose }: ToastContainerProps) {
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2">
      {toasts.map((toast) => (
        <Toast key={toast.id} toast={toast} onClose={onClose} />
      ))}
    </div>
  );
}

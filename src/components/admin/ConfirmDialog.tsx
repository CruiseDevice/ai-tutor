"use client";

import { AlertTriangle } from 'lucide-react';

interface ConfirmDialogProps {
  isOpen: boolean;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  variant?: 'danger' | 'warning' | 'info';
  onConfirm: () => void;
  onCancel: () => void;
}

export default function ConfirmDialog({
  isOpen,
  title,
  message,
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  variant = 'warning',
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  if (!isOpen) return null;

  const variantStyles = {
    danger: {
      iconBg: 'bg-danger/15',
      iconColor: 'text-danger',
      confirmButton: '',
      confirmStyle: { background: 'var(--danger)', borderColor: 'var(--danger)', color: '#fff' },
    },
    warning: {
      iconBg: 'bg-accent-soft',
      iconColor: 'text-accent',
      confirmButton: 'btn-primary',
      confirmStyle: undefined,
    },
    info: {
      iconBg: 'bg-accent-soft',
      iconColor: 'text-accent',
      confirmButton: 'btn-primary',
      confirmStyle: undefined,
    },
  };

  const styles = variantStyles[variant];

  return (
    <div className="fixed inset-0 bg-ink/40 flex items-center justify-center z-50 p-4">
      <div className="bg-paper rounded shadow-page max-w-md w-full animate-scale-in">
        <div className="p-6">
          {/* Icon */}
          <div className={`mx-auto flex items-center justify-center w-12 h-12 rounded-full ${styles.iconBg} mb-4`}>
            <AlertTriangle className={`w-6 h-6 ${styles.iconColor}`} />
          </div>

          {/* Content */}
          <div className="text-center mb-6">
            <h3 className="font-serif text-lg font-semibold text-ink mb-2">{title}</h3>
            <p className="font-serif text-sm text-subtle">{message}</p>
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <button
              onClick={onCancel}
              className="btn btn-quiet flex-1"
            >
              {cancelText}
            </button>
            <button
              onClick={onConfirm}
              className={`btn flex-1 ${styles.confirmButton}`}
              style={styles.confirmStyle}
            >
              {confirmText}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

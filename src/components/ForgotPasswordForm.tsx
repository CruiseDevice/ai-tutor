// src/components/ForgotPasswordForm.tsx
'use client'

import { CheckCircle, Copy, X } from "lucide-react";
import Link from "next/link"
import { useRouter } from "next/navigation";
import { useState } from "react";
import { authApi } from "@/lib/api-client";

export default function ForgotPasswordForm() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [resetToken, setResetToken] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setResetToken('');

    if (!email.trim()) {
      setError('Email is required');
      return;
    }

    if (!/\S+@\S+\.\S+/.test(email)) {
      setError('Please enter a valid email address');
      return;
    }

    setIsLoading(true);

    try {
      const response = await authApi.requestPasswordReset(email);

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || data.detail || 'Failed to send reset email');
      }

      // Dev mode: token is returned directly
      if (data.token) {
        setResetToken(data.token);
        setSuccess('Password reset token generated (dev mode)');
      } else {
        setSuccess('If an account with that email exists, you will receive a password reset link.');
        setEmail('');
      }
    } catch (error) {
      // TODO: Display error message to user
      console.error('Password reset request error: ', error);
      setError(error instanceof Error ? error.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopyToken = () => {
    navigator.clipboard.writeText(resetToken);
  };

  const handleResetWithToken = () => {
    router.push(`/reset-password?token=${resetToken}`);
  };
  return (
    <div className="min-h-screen flex items-center justify-center bg-paper px-4">
      <div className="w-full max-w-md">
        {/* ===== Brand wordmark ===== */}
        <header className="mb-8">
          <h1 className="font-serif text-2xl font-semibold tracking-tight">
            TUTOR<span className="text-accent">.AI</span>
          </h1>
          <p className="font-serif text-subtle mt-1">Reset your password</p>
        </header>

        {/* ===== Form card ===== */}
        <div className="surface-card p-6 sm:p-8 space-y-6">
          {/* Error Alert */}
          {error && (
            <div
              className="flex items-start gap-3 bg-accent-soft border border-hair rounded-sm px-4 py-3"
              role="alert"
            >
              <span className="text-danger font-semibold leading-none mt-0.5">!</span>
              <p className="flex-1 font-serif text-sm text-ink">{error}</p>
              <button
                onClick={() => setError('')}
                className="text-subtle hover:text-ink min-w-[44px] min-h-[44px] flex items-center justify-center transition-colors ease-desk"
                aria-label="Dismiss error"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          {/* Success Alert */}
          {success && (
            <div className="space-y-3 bg-success/10 border border-hair rounded-sm px-4 py-3" role="alert">
              <div className="flex items-start gap-3">
                <CheckCircle className="h-5 w-5 text-success flex-shrink-0 mt-0.5" />
                <p className="font-serif text-sm text-ink">{success}</p>
              </div>
              {resetToken && (
                <div className="space-y-2 pt-2 border-t border-hair-soft">
                  <p className="font-mono text-xs text-faint">Reset token (dev mode)</p>
                  <div className="flex items-center gap-2">
                    <code className="flex-1 bg-desk px-2 py-1 rounded-xs text-xs break-all font-mono text-ink-2">{resetToken}</code>
                    <button
                      type="button"
                      onClick={handleCopyToken}
                      className="btn btn-quiet px-2 py-1 min-w-[44px] min-h-[44px]"
                      title="Copy token"
                      aria-label="Copy token"
                    >
                      <Copy className="h-4 w-4" />
                    </button>
                  </div>
                  <button
                    type="button"
                    onClick={handleResetWithToken}
                    className="btn w-full"
                  >
                    Go to reset password
                  </button>
                </div>
              )}
            </div>
          )}

          <form className="space-y-5" onSubmit={handleSubmit}>
            <div className="space-y-2">
              <label htmlFor="email" className="block font-mono text-xs text-faint">
                Email
              </label>
              <input
                id="email"
                name="email"
                type="email"
                autoComplete="email"
                required
                placeholder="you@example.com"
                className="block w-full px-3 py-2.5 font-serif text-sm bg-surface border border-hair rounded-sm outline-none transition ease-desk focus:border-accent focus:ring-2 focus:ring-accent/30"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={isLoading}
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="btn btn-primary w-full disabled:cursor-not-allowed disabled:opacity-60 flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <span className="inline-block h-4 w-4 animate-spin border-2 border-paper/40 border-t-paper rounded-full" />
                  <span>Sending…</span>
                </>
              ) : (
                <span>Send reset link</span>
              )}
            </button>
          </form>

          <p className="text-center font-serif text-sm text-subtle">
            <Link href="/login" className="text-accent hover:text-accent-ink transition-colors ease-desk">
              Back to login
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}
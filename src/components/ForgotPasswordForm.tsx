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
      console.error('Password reset request error:', error);
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
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">Forgot Password</h2>
          <p className="mt-2 text-center text-sm text-gray-600">Enter your email address and we will send you a link to reset your password.</p>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
        {error && (
            <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded relative" role="alert">
              <button
                onClick={() => setError('')}
                className="absolute right-2 top-2 text-red-700 hover:text-red-900"
              >
                <X size={16} />
              </button>
              <p>{error}</p>
            </div>
          )}

          {success && (
            <div className="bg-green-100 border-l-4 border-green-500 text-green-700 p-4 rounded relative" role="alert">
              <div className="flex">
                <CheckCircle className="h-5 w-5 text-green-400" />
                <p className="ml-3">{success}</p>
              </div>
              {resetToken && (
                <div className="mt-3 space-y-2">
                  <p className="text-sm font-medium">Reset Token (Dev Mode):</p>
                  <div className="flex items-center gap-2">
                    <code className="flex-1 bg-white px-2 py-1 rounded text-xs break-all">{resetToken}</code>
                    <button
                      type="button"
                      onClick={handleCopyToken}
                      className="p-1 hover:bg-green-200 rounded"
                      title="Copy token"
                    >
                      <Copy size={16} />
                    </button>
                  </div>
                  <button
                    type="button"
                    onClick={handleResetWithToken}
                    className="w-full mt-2 py-1 px-3 bg-green-600 hover:bg-green-700 text-white rounded text-sm"
                  >
                    Go to Reset Password
                  </button>
                </div>
              )}
            </div>
          )}
          <div>
            <label htmlFor="email" className="sr-only">Email address</label>
            <input
              id="email"
              name="email"
              type="email"
              autoComplete="email"
              required
              placeholder="Email address"
              className="appearance-none rounded-none relative-block block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 focus:z-10 sm:text-sm"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={isLoading}
            />
          </div>
          <div>
            <button
              type="submit"
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50">
                {isLoading ? 'Sending...' : 'Send Reset Link'}
            </button>
          </div>
          <div className="text-sm text-center">
            <Link href="/login" className="font-medium text-indigo-600 hover:text-indigo-500">
              Back to login
            </Link>
          </div>
        </form>
      </div>
    </div>
  )
}
// src/components/ResetPasswordForm.tsx
'use client'

import { CheckCircle, X } from "lucide-react";
import Link from "next/link"
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { authApi } from "@/lib/api-client";

interface FormErrors {
  password?: string;
  confirmPassword?: string;
  general?: string;
}

function ResetPasswordFormWithParams() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [errors, setErrors] = useState<FormErrors>({})
  const [isLoading, setIsLoading] = useState(false);
  const [success, setSuccess] = useState('');
  const [token, setToken] = useState('');

  const [formData, setFormData] = useState({
    password: '',
    confirmPassword: ''
  })

  useEffect(() => {
    const tokenParam = searchParams.get('token');
    if (!tokenParam) {
      setErrors(prev => ({
        ...prev,
        general: 'Invalid or missing reset token'
      }));
      router.push('/forgot-password');
    } else {
      setToken(tokenParam);
    }
  }, [searchParams, router]);

  const validateForm = () => {
    const newErrors: FormErrors = {};
    if (!formData.password) {
      newErrors.password = 'Password is required';
    }

    if (!formData.confirmPassword) {
      newErrors.confirmPassword = 'Please confirm your password';
    } else if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const {name, value} = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  }

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setErrors({});
    setSuccess('');

    if(!validateForm()) return;

    if (!token) {
      setErrors(prev => ({
        ...prev,
        general: 'Invalid or missing reset token'
      }));
      return;
    }

    setIsLoading(true);

    try {
      const response = await authApi.confirmPasswordReset(token, formData.password);

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || data.detail || 'Failed to reset password');
      }

      setSuccess('Your password has been reset successfully. You can now log in with your new password.');
      setFormData({password: '', confirmPassword: ''});

    } catch (error) {
      setErrors(prev => ({
        ...prev,
        general: error instanceof Error ? error.message : 'An error occurred while resetting your password'
      }));
    } finally {
      setIsLoading(false);
    }
  };

  if (!token) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-paper px-4">
        <div className="w-full max-w-md text-center">
          <p className="font-serif text-danger">Invalid or missing reset token.</p>
          <Link href="/forgot-password" className="font-serif text-sm text-accent hover:text-accent-ink transition-colors ease-desk mt-2 inline-block">
            Request a new password reset
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-paper px-4">
      <div className="w-full max-w-md">
        {/* ===== Brand wordmark ===== */}
        <header className="mb-8">
          <h1 className="font-serif text-2xl font-semibold tracking-tight">
            TUTOR<span className="text-accent">.AI</span>
          </h1>
          <p className="font-serif text-subtle mt-1">Enter your new password</p>
        </header>

        {/* ===== Form card ===== */}
        <div className="surface-card p-6 sm:p-8 space-y-6">
          {errors.general && (
            <div
              className="flex items-start gap-3 bg-accent-soft border border-hair rounded-sm px-4 py-3"
              role="alert"
            >
              <span className="text-danger font-semibold leading-none mt-0.5">!</span>
              <p className="flex-1 font-serif text-sm text-ink">{errors.general}</p>
              <button
                onClick={() => setErrors(prev => ({ ...prev, general: undefined }))}
                className="text-subtle hover:text-ink min-w-[44px] min-h-[44px] flex items-center justify-center transition-colors ease-desk"
                aria-label="Dismiss error"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          )}

          {success && (
            <div className="flex items-start gap-3 bg-success/10 border border-hair rounded-sm px-4 py-3" role="alert">
              <CheckCircle className="h-5 w-5 text-success flex-shrink-0 mt-0.5" />
              <p className="font-serif text-sm text-ink">{success}</p>
            </div>
          )}

          <form className="space-y-5" onSubmit={handleSubmit}>
            <div className="space-y-2">
              <label htmlFor="password" className="block font-mono text-xs text-faint">
                New password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                required
                placeholder="••••••••"
                value={formData.password}
                onChange={handleChange}
                className={`block w-full px-3 py-2.5 font-serif text-sm bg-surface border rounded-sm outline-none transition ease-desk focus:ring-2 focus:ring-accent/30 ${
                  errors.password ? 'border-danger' : 'border-hair focus:border-accent'
                }`}
                disabled={isLoading}
              />
              {errors.password && (
                <p className="font-serif text-xs text-danger">{errors.password}</p>
              )}
            </div>

            <div className="space-y-2">
              <label htmlFor="confirmPassword" className="block font-mono text-xs text-faint">
                Confirm password
              </label>
              <input
                id="confirmPassword"
                name="confirmPassword"
                type="password"
                required
                placeholder="••••••••"
                value={formData.confirmPassword}
                onChange={handleChange}
                className={`block w-full px-3 py-2.5 font-serif text-sm bg-surface border rounded-sm outline-none transition ease-desk focus:ring-2 focus:ring-accent/30 ${
                  errors.confirmPassword ? 'border-danger' : 'border-hair focus:border-accent'
                }`}
                disabled={isLoading}
              />
              {errors.confirmPassword && (
                <p className="font-serif text-xs text-danger">{errors.confirmPassword}</p>
              )}
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="btn btn-primary w-full disabled:cursor-not-allowed disabled:opacity-60 flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <span className="inline-block h-4 w-4 animate-spin border-2 border-paper/40 border-t-paper rounded-full" />
                  <span>Resetting…</span>
                </>
              ) : (
                <span>Reset password</span>
              )}
            </button>
          </form>

          <p className="text-center font-serif text-sm text-subtle">
            <Link
              href="/login"
              className="text-accent hover:text-accent-ink transition-colors ease-desk"
            >
              Back to login
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}


// main export with Suspense wrapper
export default function ResetPasswordForm() {
  return (
    <Suspense fallback={<div>Loading ...</div>}>
      <ResetPasswordFormWithParams />
    </Suspense>
  )
}
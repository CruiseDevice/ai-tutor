// app/components/LoginForm.tsx
'use client';

import Link from "next/link";
import { useRouter } from "next/navigation";
import React, { useState } from "react";
import { X } from "lucide-react";
import { authApi } from "@/lib/api-client";

interface FormErrors {
  email?: string;
  password?: string;
  general?: string;
}

export default function LoginForm() {
  const router = useRouter();
  const [errors, setErrors] = useState<FormErrors>({});
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });

  const validateForm = (): boolean => {
    const newErrors: FormErrors = {};

    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Please enter a valid email address';
    }

    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const {name, value} = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear error for the field being edited
    if (errors[name as keyof FormErrors]) {
      setErrors(prev => ({
        ...prev,
        [name]: undefined
      }));
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setErrors({});

    if (!validateForm()) {
      return;
    }

    setLoading(true);

    try {
      const response = await authApi.login(formData.email, formData.password);

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        if (response.status === 401) {
          throw new Error('Invalid email or password');
        } else if (response.status === 429) {
          throw new Error('Too many login attempts. Please try again later');
        } else {
          throw new Error(data.error || data.detail || 'Login failed. Please try again');
        }
      }

      // Login successful - redirect to dashboard
      router.push('/dashboard');
    } catch (error) {
      if (error instanceof Error) {
        setErrors(prev => ({
          ...prev,
          general: error.message
        }));
      } else {
        setErrors(prev => ({
          ...prev,
          general: 'An unexpected error occurred. Please try again'
        }));
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50 px-4">
      <div className="w-full max-w-md">
        <header className="mb-6">
          <p className="text-[11px] font-semibold tracking-[0.2em] text-indigo-600 uppercase">
            StudyFetch
          </p>
          <h1 className="mt-2 text-2xl font-semibold text-slate-900">
            Sign in to continue
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            Pick up where you left off with your tutor.
          </p>
        </header>

        <div className="rounded-xl bg-white border border-slate-200 shadow-sm p-6 sm:p-7 space-y-6">
          {errors.general && (
            <div
              className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-3.5 py-3 text-sm text-red-700"
              role="alert"
            >
              <div className="mt-0.5 flex-1">
                <p className="font-medium">Couldn&apos;t sign you in</p>
                <p className="mt-0.5 text-xs sm:text-sm">{errors.general}</p>
              </div>
              <button
                type="button"
                onClick={() => setErrors(prev => ({ ...prev, general: undefined }))}
                className="shrink-0 text-red-500 hover:text-red-700"
              >
                <X size={16} />
              </button>
            </div>
          )}

          <form className="space-y-5" onSubmit={handleSubmit}>
            <div className="space-y-1.5">
              <label
                htmlFor="email"
                className="block text-sm font-medium text-slate-700"
              >
                Email
              </label>
              <input
                id="email"
                name="email"
                type="email"
                autoComplete="email"
                required
                placeholder="you@example.com"
                className={`block w-full rounded-md border px-3 py-2.5 text-sm shadow-sm outline-none transition focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 ${
                  errors.email
                    ? "border-red-500 focus:ring-red-500 focus:border-red-500"
                    : "border-slate-300"
                }`}
                value={formData.email}
                onChange={handleChange}
              />
              {errors.email && (
                <p className="mt-1 text-xs text-red-600">{errors.email}</p>
              )}
            </div>

            <div className="space-y-1.5">
              <label
                htmlFor="password"
                className="block text-sm font-medium text-slate-700"
              >
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                required
                placeholder="••••••••"
                className={`block w-full rounded-md border px-3 py-2.5 text-sm shadow-sm outline-none transition focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 ${
                  errors.password
                    ? "border-red-500 focus:ring-red-500 focus:border-red-500"
                    : "border-slate-300"
                }`}
                value={formData.password}
                onChange={handleChange}
              />
              {errors.password && (
                <p className="mt-1 text-xs text-red-600">{errors.password}</p>
              )}
            </div>

            <div className="flex items-center justify-between text-xs sm:text-sm">
              <label className="inline-flex items-center gap-2 text-slate-600">
                <input
                  type="checkbox"
                  className="h-3.5 w-3.5 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                />
                <span>Keep me signed in</span>
              </label>
              <Link
                href="/forgot-password"
                className="font-medium text-indigo-600 hover:text-indigo-500"
              >
                Forgot password?
              </Link>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading && (
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
              )}
              <span>{loading ? "Signing you in..." : "Sign in"}</span>
            </button>
          </form>

          <p className="text-center text-xs sm:text-sm text-slate-500">
            Don&apos;t have an account?{" "}
            <Link
              href="/register"
              className="font-medium text-indigo-600 hover:text-indigo-500"
            >
              Create a StudyFetch account
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}

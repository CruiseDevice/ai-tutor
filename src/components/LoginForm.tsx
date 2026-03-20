// app/components/LoginForm.tsx
'use client';

import Link from "next/link";
import { useRouter } from "next/navigation";
import React, { useState } from "react";
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
    <div className="min-h-screen flex items-center justify-center bg-paper px-4">
      <div className="w-full max-w-md">
        {/* =====================================================
            HEADER - TUTOR.AI branding
            ===================================================== */}
        <header className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h1 className="font-mono font-bold text-2xl uppercase tracking-tight">
              TUTOR<span className="text-accent">.AI</span>
            </h1>
            <span className="font-mono text-xs text-accent">[001]</span>
          </div>
          <p className="font-serif text-ink">
            Sign in to continue
          </p>
        </header>

        {/* =====================================================
            FORM CARD - Brutalist Style
            ===================================================== */}
        <div className="bg-panel-bg border-2 border-ink p-6 sm:p-8 space-y-6">
          {/* Error Alert */}
          {errors.general && (
            <div
              className="flex items-start gap-3 border-2 border-accent bg-accent/10 px-4 py-3"
              role="alert"
            >
              <span className="font-mono text-accent text-lg">[!]</span>
              <div className="flex-1">
                <p className="font-mono text-xs font-bold text-accent uppercase">Couldn&apos;t sign you in</p>
                <p className="font-serif text-sm text-ink mt-1">{errors.general}</p>
              </div>
              <button
                type="button"
                onClick={() => setErrors(prev => ({ ...prev, general: undefined }))}
                className="font-mono text-accent hover:text-ink min-w-[44px] min-h-[44px] flex items-center justify-center"
              >
                [×]
              </button>
            </div>
          )}

          <form className="space-y-5" onSubmit={handleSubmit}>
            {/* Email Field */}
            <div className="space-y-2">
              <label
                htmlFor="email"
                className="block font-mono text-xs uppercase tracking-wider text-ink"
              >
                Email
              </label>
              <input
                id="email"
                name="email"
                type="email"
                autoComplete="email"
                required
                placeholder="[you@example.com]"
                className={`block w-full px-3 py-2.5 font-serif text-sm bg-paper border-2 outline-none transition focus:ring-2 ${
                  errors.email
                    ? "border-accent ring-accent/50"
                    : "border-ink focus:ring-accent/50"
                }`}
                value={formData.email}
                onChange={handleChange}
              />
              {errors.email && (
                <p className="font-mono text-xs text-accent">[{errors.email}]</p>
              )}
            </div>

            {/* Password Field */}
            <div className="space-y-2">
              <label
                htmlFor="password"
                className="block font-mono text-xs uppercase tracking-wider text-ink"
              >
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                required
                placeholder="[••••••••]"
                className={`block w-full px-3 py-2.5 font-serif text-sm bg-paper border-2 outline-none transition focus:ring-2 ${
                  errors.password
                    ? "border-accent ring-accent/50"
                    : "border-ink focus:ring-accent/50"
                }`}
                value={formData.password}
                onChange={handleChange}
              />
              {errors.password && (
                <p className="font-mono text-xs text-accent">[{errors.password}]</p>
              )}
            </div>

            {/* Remember & Forgot */}
            <div className="flex items-center justify-between font-mono text-xs">
              <label className="inline-flex items-center gap-2 text-ink">
                <input
                  type="checkbox"
                  className="w-4 h-4 accent-accent"
                />
                <span>[Keep me signed in]</span>
              </label>
              <Link
                href="/forgot-password"
                className="text-accent hover:underline"
              >
                [Forgot password?]
              </Link>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="brutalist-button brutalist-button-primary w-full py-3 font-mono text-sm uppercase disabled:cursor-not-allowed disabled:opacity-60 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <span className="inline-block h-4 w-4 animate-spin border-2 border-paper/40 border-t-paper rounded-full" />
                  <span>[Signing in...]</span>
                </>
              ) : (
                <span>[Sign in]</span>
              )}
            </button>
          </form>

          {/* Register Link */}
          <p className="text-center font-serif text-sm text-subtle">
            Don&apos;t have an account?{" "}
            <Link
              href="/register"
              className="font-mono text-xs text-accent hover:underline"
            >
              [Create a TUTOR.AI account]
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}

// app/components/RegisterForm.tsx
'use client';

import Link from "next/link";
import React, { useState } from "react"
import { useRouter } from "next/navigation";
import { authApi } from "@/lib/api-client";

interface FormErrors {
  email?: string;
  password?: string;
  confirmPassword?: string;
  general?: string;
}

export default function RegisterForm () {
  const router = useRouter();
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword:'',
  });
  const [errors, setErrors] = useState<FormErrors>({});
  const [loading, setLoading] = useState(false);

  const validateForm = (): boolean => {
    const newErrors: FormErrors = {};

    // Email validation
    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Please enter a valid email address';
    }

    // Password validation
    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 8) {
      newErrors.password = 'Password must be at least 8 characters long';
    }

    // Confirm password validation
    if (!formData.confirmPassword) {
      newErrors.confirmPassword = 'Please confirm your password';
    } else if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
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

  const handleSubmit = async(e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setErrors({});

    if (!validateForm()) {
      return;
    }

    setLoading(true);

    try {
      const response = await authApi.register(formData.email, formData.password);

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        if (response.status === 409 || response.status === 400) {
          throw new Error(data.detail || data.error || 'Email already registered or invalid data');
        } else {
          throw new Error(data.detail || data.error || 'Registration failed. Please try again');
        }
      }

      // Registration successful - redirect to dashboard
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
            <span className="font-mono text-xs text-accent">[002]</span>
          </div>
          <p className="font-serif text-ink">
            Create your account
          </p>
        </header>

        {/* =====================================================
            FORM CARD - Brutalist Style
            ===================================================== */}
        <div className="bg-panel-bg border-2 border-ink p-6 sm:p-8 space-y-5">
          {/* Error Alert */}
          {errors.general && (
            <div
              className="flex items-start gap-3 border-2 border-accent bg-accent/10 px-4 py-3"
              role="alert"
            >
              <span className="font-mono text-accent text-lg">[!]</span>
              <div className="flex-1">
                <p className="font-mono text-xs font-bold text-accent uppercase">Couldn&apos;t create account</p>
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

          {/* Info Box - Password Requirements */}
          <div className="border-2 border-ink bg-accent/5 px-4 py-3">
            <div className="flex items-start gap-2">
              <span className="font-mono text-accent text-sm">[ℹ]</span>
              <div>
                <p className="font-mono text-xs font-bold text-ink uppercase">Password requirements</p>
                <ul className="mt-1.5 font-serif text-sm text-subtle">
                  <li>• At least 8 characters</li>
                </ul>
              </div>
            </div>
          </div>

          <form className="space-y-4" onSubmit={handleSubmit}>
            {/* Email Field */}
            <div className="space-y-2">
              <label
                htmlFor="email"
                className="block font-mono text-xs uppercase tracking-wider text-ink"
              >
                Email
              </label>
              <input
                type="email"
                id="email"
                name="email"
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
                type="password"
                id="password"
                name="password"
                autoComplete="new-password"
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

            {/* Confirm Password Field */}
            <div className="space-y-2">
              <label
                htmlFor="confirmPassword"
                className="block font-mono text-xs uppercase tracking-wider text-ink"
              >
                Confirm password
              </label>
              <input
                type="password"
                id="confirmPassword"
                name="confirmPassword"
                autoComplete="new-password"
                required
                placeholder="[Repeat your password]"
                className={`block w-full px-3 py-2.5 font-serif text-sm bg-paper border-2 outline-none transition focus:ring-2 ${
                  errors.confirmPassword
                    ? "border-accent ring-accent/50"
                    : "border-ink focus:ring-accent/50"
                }`}
                value={formData.confirmPassword}
                onChange={handleChange}
              />
              {errors.confirmPassword && (
                <p className="font-mono text-xs text-accent">
                  [{errors.confirmPassword}]
                </p>
              )}
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
                  <span>[Creating account...]</span>
                </>
              ) : (
                <span>[Sign up]</span>
              )}
            </button>
          </form>

          {/* Login Link */}
          <p className="pt-1 text-center font-serif text-sm text-subtle">
            Already have an account?{" "}
            <Link
              href="/login"
              className="font-mono text-xs text-accent hover:underline"
            >
              [Sign in]
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}

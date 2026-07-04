'use client';

import { useRouter } from "next/navigation";
import React, { useState } from "react";
import { ArrowLeft, Check, KeyRound, X } from "lucide-react";
import { userApi } from "@/lib/api-client";

export default function APISettings() {
  const router = useRouter();
  const [apiKey, setApiKey] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setSuccess('');

    try {
      const response = await userApi.updateAPIKey(apiKey);

      if(!response.ok) {
        const data = await response.json();
        throw new Error(data.error || data.detail || 'Failed to save API key');
      }

      setSuccess('API key saved successfully');
      setApiKey('');

      // Clear success message after 5 seconds
      setTimeout(() => setSuccess(''), 5000);
    } catch (error) {
      setError(error instanceof Error? error.message : 'Failed to save API key');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-paper p-4 sm:p-6 lg:p-8">
      <div className="max-w-2xl mx-auto">
        {/* ===== Back link ===== */}
        <button
          onClick={() => router.back()}
          className="btn btn-quiet mb-8 min-h-[44px]"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to dashboard
        </button>

        {/* ===== Main card ===== */}
        <div className="surface-card overflow-hidden">
          {/* Header */}
          <div className="border-b border-hair p-6 sm:p-8">
            <div className="flex items-start gap-4">
              <div className="w-11 h-11 rounded-sm bg-accent-soft flex items-center justify-center flex-shrink-0">
                <KeyRound className="h-5 w-5 text-accent" />
              </div>
              <div className="flex-1">
                <h1 className="font-serif text-xl sm:text-2xl font-semibold tracking-tight">API Settings</h1>
                <p className="font-serif text-sm text-subtle leading-relaxed mt-1">
                  Configure your OpenAI API key to enable personalized chat completions
                </p>
              </div>
            </div>
          </div>

          {/* Form Section */}
          <div className="p-6 sm:p-8">
            {/* Error Message */}
            {error && (
              <div
                className="mb-6 flex items-start gap-3 bg-accent-soft border border-hair rounded-sm px-4 py-3"
                role="alert"
              >
                <span className="text-danger font-semibold leading-none mt-0.5">!</span>
                <div className="flex-1">
                  <p className="font-serif text-xs font-semibold text-danger">Error</p>
                  <p className="font-serif text-sm text-ink mt-1">{error}</p>
                </div>
                <button
                  onClick={() => setError('')}
                  className="text-subtle hover:text-ink min-w-[44px] min-h-[44px] flex items-center justify-center transition-colors ease-desk"
                  aria-label="Dismiss error"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            )}

            {/* Success Message */}
            {success && (
              <div
                className="mb-6 flex items-start gap-3 bg-success/10 border border-hair rounded-sm px-4 py-3"
                role="alert"
              >
                <Check className="h-5 w-5 text-success flex-shrink-0 mt-0.5" />
                <p className="flex-1 font-serif text-sm text-ink">{success}</p>
                <button
                  onClick={() => setSuccess('')}
                  className="text-subtle hover:text-ink min-w-[44px] min-h-[44px] flex items-center justify-center transition-colors ease-desk"
                  aria-label="Dismiss"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* API Key Input */}
              <div className="space-y-2">
                <label
                  htmlFor="apiKey"
                  className="block font-mono text-xs text-faint"
                >
                  OpenAI API key
                </label>

                <input
                  type="password"
                  id="apiKey"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-proj-…"
                  className="w-full px-3 py-2.5 font-serif text-sm bg-surface border border-hair rounded-sm outline-none transition ease-desk focus:border-accent focus:ring-2 focus:ring-accent/30 placeholder:text-faint placeholder:italic"
                />

                {/* Info Card */}
                <div className="bg-desk border border-hair rounded-sm px-4 py-3 mt-3">
                  <p className="font-serif text-sm text-ink leading-relaxed">
                    Enter your OpenAI API key to use your own account for chat completion. Your key is encrypted and stored securely.
                  </p>
                  <a
                    href="https://platform.openai.com/api-keys"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 mt-2 font-serif text-sm text-accent hover:text-accent-ink transition-colors ease-desk"
                  >
                    Get your API key →
                  </a>
                </div>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isLoading || !apiKey.trim()}
                className="btn btn-primary w-full disabled:cursor-not-allowed disabled:opacity-60 flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <span className="inline-block h-4 w-4 animate-spin border-2 border-paper/40 border-t-paper rounded-full" />
                    <span>Saving…</span>
                  </>
                ) : (
                  <span>Save API key</span>
                )}
              </button>
            </form>

            {/* Security Notice */}
            <div className="mt-8 pt-6 border-t border-hair">
              <div className="flex items-start gap-3 bg-desk border border-hair rounded-sm px-4 py-3">
                <span className="text-accent font-semibold leading-none mt-0.5">⚠</span>
                <div>
                  <p className="font-mono text-xs text-faint mb-1">Security notice</p>
                  <p className="font-serif text-xs text-ink leading-relaxed">
                    Your API key is encrypted and stored securely. Never share your API key with others or commit it to version control.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

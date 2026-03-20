'use client';

import { useRouter } from "next/navigation";
import React, { useState } from "react";
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
      {/* Background Pattern - Subtle */}
      <div className="fixed inset-0 z-0 opacity-[0.02] pointer-events-none"
        style={{
          backgroundImage: 'radial-gradient(#0a0a0a 1px, transparent 1px)',
          backgroundSize: '24px 24px'
        }}
      />

      <div className="max-w-2xl mx-auto relative z-10">
        {/* =====================================================
            BACK BUTTON - Brutalist Style
            ===================================================== */}
        <button
          onClick={() => router.back()}
          className="mb-8 font-mono text-xs px-4 py-2.5 border border-ink hover:bg-ink hover:text-paper transition-colors flex items-center gap-2 min-w-[44px] min-h-[44px]"
        >
          [←] Back to Dashboard
        </button>

        {/* =====================================================
            MAIN CARD - Brutalist Style
            ===================================================== */}
        <div className="bg-panel-bg border-2 border-ink overflow-hidden">
          {/* Header Section - No Gradient, Stark Style */}
          <div className="border-b-2 border-ink p-6 sm:p-8">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 border-2 border-ink flex items-center justify-center font-mono text-2xl bg-paper">
                [⚡]
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <h1 className="font-mono text-xl sm:text-2xl font-bold uppercase">API Settings</h1>
                  <span className="font-mono text-xs text-accent">[003]</span>
                </div>
                <p className="font-serif text-sm text-subtle leading-relaxed">
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
                className="mb-6 border-2 border-accent bg-accent/10 px-4 py-3 flex items-start gap-3"
                role="alert"
              >
                <span className="font-mono text-accent text-lg">[!]</span>
                <div className="flex-1">
                  <p className="font-mono text-xs font-bold text-accent uppercase">Error</p>
                  <p className="font-serif text-sm text-ink mt-1">{error}</p>
                </div>
                <button
                  onClick={() => setError('')}
                  className="font-mono text-accent hover:text-ink min-w-[44px] min-h-[44px] flex items-center justify-center"
                >
                  [×]
                </button>
              </div>
            )}

            {/* Success Message */}
            {success && (
              <div
                className="mb-6 border-2 border-ink bg-ink text-paper px-4 py-3 flex items-start gap-3"
                role="alert"
              >
                <span className="font-mono text-lg">[✓]</span>
                <div className="flex-1">
                  <p className="font-serif text-sm">{success}</p>
                </div>
                <button
                  onClick={() => setSuccess('')}
                  className="font-mono hover:underline min-w-[44px] min-h-[44px] flex items-center justify-center"
                >
                  [×]
                </button>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* API Key Input */}
              <div className="space-y-3">
                <label
                  htmlFor="apiKey"
                  className="flex items-center gap-2 font-mono text-xs uppercase tracking-wider text-ink"
                >
                  <span>[⚡]</span>
                  <span>OpenAI API Key</span>
                </label>

                <input
                  type="password"
                  id="apiKey"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="[sk-proj-...]"
                  className="w-full px-4 py-3.5 font-serif text-sm bg-paper border-2 border-ink focus:outline-none focus:ring-2 focus:ring-accent/50 placeholder:text-subtle"
                />

                {/* Info Card */}
                <div className="border-2 border-ink bg-accent/5 px-4 py-3">
                  <div className="flex items-start gap-3">
                    <span className="font-mono text-accent text-sm">[ℹ]</span>
                    <div className="flex-1">
                      <p className="font-serif text-sm text-ink leading-relaxed">
                        Enter your OpenAI API key to use your own account for chat completion. Your key is encrypted and stored securely.
                      </p>
                      <a
                        href="https://platform.openai.com/api-keys"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 mt-2 font-mono text-xs text-accent hover:underline"
                      >
                        [Get your API key →]
                      </a>
                    </div>
                  </div>
                </div>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isLoading || !apiKey.trim()}
                className={`w-full py-3.5 px-6 font-mono text-sm uppercase border-2 flex items-center justify-center gap-2 min-h-[44px] transition-all duration-150 ${
                  isLoading || !apiKey.trim()
                    ? 'bg-paper text-subtle border-ink cursor-not-allowed'
                    : 'bg-ink text-paper border-ink hover:bg-accent hover:border-accent'
                }`}
              >
                {isLoading ? (
                  <>
                    <span className="inline-block h-4 w-4 animate-spin border-2 border-paper/40 border-t-paper rounded-full" />
                    <span>[Saving...]</span>
                  </>
                ) : (
                  <>
                    <span>[✓]</span>
                    <span>Save API Key</span>
                  </>
                )}
              </button>
            </form>

            {/* Security Notice */}
            <div className="mt-8 pt-6 border-t-2 border-ink">
              <div className="flex items-start gap-3 border-2 border-ink bg-accent/5 px-4 py-3">
                <span className="font-mono text-accent text-lg">[⚠]</span>
                <div>
                  <p className="font-mono text-xs font-bold text-accent uppercase mb-1">Security Notice</p>
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

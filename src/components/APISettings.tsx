'use client';

import { ArrowLeft, Key, Shield, CheckCircle2, AlertCircle, Loader2, Sparkles } from "lucide-react";
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
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-indigo-50/20 p-4 sm:p-6 lg:p-8">
      {/* Background Pattern */}
      <div className="fixed inset-0 z-0 opacity-[0.03]"
        style={{
          backgroundImage: 'radial-gradient(#64748b 1px, transparent 1px)',
          backgroundSize: '32px 32px'
        }}
      />

      <div className="max-w-2xl mx-auto relative z-10">
        {/* Back Button */}
        <button
          onClick={() => router.back()}
          className="mb-8 group flex items-center gap-2 px-4 py-2.5 bg-white/80 backdrop-blur-sm border border-gray-200 rounded-xl shadow-sm hover:shadow-md hover:border-blue-300 transition-all duration-200 text-sm font-medium text-slate-700 hover:text-blue-600"
        >
          <ArrowLeft size={18} className="group-hover:-translate-x-0.5 transition-transform duration-200"/>
          <span>Back to Dashboard</span>
        </button>

        {/* Main Card */}
        <div className="bg-white/90 backdrop-blur-md rounded-3xl shadow-xl border border-gray-100 overflow-hidden">
          {/* Header Section with Gradient */}
          <div className="relative bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600 p-8 sm:p-10">
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmZmZmYiIGZpbGwtb3BhY2l0eT0iMC4xIj48Y2lyY2xlIGN4PSIzMCIgY3k9IjMwIiByPSIyIi8+PC9nPjwvZz48L3N2Zz4=')] opacity-20"></div>
            <div className="relative z-10 flex items-start gap-4">
              <div className="w-14 h-14 rounded-2xl bg-white/20 backdrop-blur-sm flex items-center justify-center shadow-lg border border-white/30">
                <Key size={28} className="text-white" />
              </div>
              <div className="flex-1">
                <h1 className="text-3xl sm:text-4xl font-bold text-white mb-2">API Settings</h1>
                <p className="text-blue-100 text-sm sm:text-base leading-relaxed">
                  Configure your OpenAI API key to enable personalized chat completions
                </p>
              </div>
            </div>
          </div>

          {/* Form Section */}
          <div className="p-6 sm:p-8 lg:p-10">
            {/* Error Message */}
            {error && (
              <div
                className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl flex items-start gap-3 animate-in fade-in slide-in-from-top-2 duration-300"
                role="alert"
              >
                <div className="p-1.5 bg-red-100 rounded-lg flex-shrink-0">
                  <AlertCircle size={18} className="text-red-600"/>
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-red-800">{error}</p>
                </div>
                <button
                  onClick={() => setError('')}
                  className="text-red-400 hover:text-red-600 transition-colors"
                >
                  <span className="sr-only">Close</span>
                  ×
                </button>
              </div>
            )}

            {/* Success Message */}
            {success && (
              <div
                className="mb-6 p-4 bg-green-50 border border-green-200 rounded-xl flex items-start gap-3 animate-in fade-in slide-in-from-top-2 duration-300"
                role="alert"
              >
                <div className="p-1.5 bg-green-100 rounded-lg flex-shrink-0">
                  <CheckCircle2 size={18} className="text-green-600"/>
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-green-800">{success}</p>
                </div>
                <button
                  onClick={() => setSuccess('')}
                  className="text-green-400 hover:text-green-600 transition-colors"
                >
                  <span className="sr-only">Close</span>
                  ×
                </button>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* API Key Input */}
              <div className="space-y-3">
                <label
                  htmlFor="apiKey"
                  className="flex items-center gap-2 text-sm font-semibold text-slate-700"
                >
                  <Shield size={16} className="text-indigo-500"/>
                  <span>OpenAI API Key</span>
                </label>

                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 via-indigo-500/10 to-purple-500/10 rounded-xl blur-sm"></div>
                  <input
                    type="password"
                    id="apiKey"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="sk-proj-..."
                    className="relative w-full px-4 py-3.5 bg-white border-2 border-gray-200 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all duration-200 text-sm text-slate-800 placeholder:text-slate-400"
                  />
                </div>

                {/* Info Card */}
                <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl">
                  <div className="flex items-start gap-3">
                    <div className="p-1.5 bg-blue-100 rounded-lg flex-shrink-0 mt-0.5">
                      <Sparkles size={14} className="text-blue-600"/>
                    </div>
                    <div className="flex-1">
                      <p className="text-xs sm:text-sm text-slate-600 leading-relaxed">
                        Enter your OpenAI API key to use your own account for chat completion. Your key is encrypted and stored securely.
                      </p>
                      <a
                        href="https://platform.openai.com/api-keys"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 mt-2 text-xs font-medium text-blue-600 hover:text-blue-700 transition-colors"
                      >
                        Get your API key
                        <ArrowLeft size={12} className="rotate-180"/>
                      </a>
                    </div>
                  </div>
                </div>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isLoading || !apiKey.trim()}
                className={`w-full flex items-center justify-center gap-2 py-3.5 px-6 rounded-xl shadow-lg font-medium text-sm transition-all duration-200 ${
                  isLoading || !apiKey.trim()
                    ? 'bg-slate-200 text-slate-400 cursor-not-allowed shadow-none'
                    : 'bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white hover:shadow-xl hover:scale-[1.02] active:scale-[0.98]'
                }`}
              >
                {isLoading ? (
                  <>
                    <Loader2 size={18} className="animate-spin"/>
                    <span>Saving...</span>
                  </>
                ) : (
                  <>
                    <CheckCircle2 size={18}/>
                    <span>Save API Key</span>
                  </>
                )}
              </button>
            </form>

            {/* Security Note */}
            <div className="mt-8 pt-6 border-t border-gray-200">
              <div className="flex items-start gap-3 p-4 bg-amber-50 border border-amber-200 rounded-xl">
                <Shield size={18} className="text-amber-600 flex-shrink-0 mt-0.5"/>
                <div>
                  <p className="text-xs font-semibold text-amber-900 mb-1">Security Notice</p>
                  <p className="text-xs text-amber-700 leading-relaxed">
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
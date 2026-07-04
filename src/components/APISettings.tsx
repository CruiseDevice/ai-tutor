'use client';

import { useRouter } from "next/navigation";
import React, { useState } from "react";
import { ArrowLeft, Check, KeyRound, X } from "lucide-react";
import { userApi } from "@/lib/api-client";

type ProviderKey = 'openai' | 'anthropic' | 'ollama';

interface ProviderConfig {
  key: ProviderKey;
  label: string;
  title: string;
  blurb: string;
  placeholder: string;
  helpUrl: string;
  helpText: string;
}

const PROVIDERS: ProviderConfig[] = [
  {
    key: 'openai',
    label: 'OpenAI',
    title: 'OpenAI API key',
    blurb: 'Powers GPT-5.1 and GPT-4o mini. Keys start with "sk-".',
    placeholder: 'sk-proj-…',
    helpUrl: 'https://platform.openai.com/api-keys',
    helpText: 'Get your OpenAI API key',
  },
  {
    key: 'anthropic',
    label: 'Anthropic',
    title: 'Anthropic API key',
    blurb: 'Powers Claude Sonnet and Haiku. Keys start with "sk-ant-".',
    placeholder: 'sk-ant-…',
    helpUrl: 'https://console.anthropic.com/settings/keys',
    helpText: 'Get your Anthropic API key',
  },
  {
    key: 'ollama',
    label: 'Ollama Cloud',
    title: 'Ollama Cloud API key',
    blurb: 'Powers open-weight models hosted on Ollama Cloud.',
    placeholder: 'your-ollama-cloud-key',
    helpUrl: 'https://ollama.com/login',
    helpText: 'Get your Ollama Cloud key',
  },
];

export default function APISettings() {
  const router = useRouter();
  const [keys, setKeys] = useState<Record<ProviderKey, string>>({
    openai: '',
    anthropic: '',
    ollama: '',
  });
  const [configured, setConfigured] = useState<Record<ProviderKey, boolean>>({
    openai: false,
    anthropic: false,
    ollama: false,
  });
  const [loadingProvider, setLoadingProvider] = useState<ProviderKey | null>(null);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  // Fetch current key status on mount
  React.useEffect(() => {
    (async () => {
      try {
        const res = await userApi.checkAPIKey();
        if (res.ok) {
          const data = await res.json();
          setConfigured({
            openai: !!data.openai,
            anthropic: !!data.anthropic,
            ollama: !!data.ollama,
          });
        }
      } catch {
        // Non-fatal: user can still save keys
      }
    })();
  }, []);

  const handleSave = async (provider: ProviderKey) => {
    const value = keys[provider].trim();
    if (!value) return;
    setLoadingProvider(provider);
    setError('');
    setSuccess('');

    try {
      const response = await userApi.updateAPIKey(provider, value);
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || data.detail || 'Failed to save API key');
      }
      setSuccess(`${PROVIDERS.find(p => p.key === provider)?.label} API key saved successfully`);
      setKeys(k => ({ ...k, [provider]: '' }));
      setConfigured(c => ({ ...c, [provider]: true }));
      setTimeout(() => setSuccess(''), 5000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save API key');
    } finally {
      setLoadingProvider(null);
    }
  };

  const handleDelete = async (provider: ProviderKey) => {
    setLoadingProvider(provider);
    setError('');
    setSuccess('');
    try {
      const response = await userApi.deleteAPIKey(provider);
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || data.detail || 'Failed to delete API key');
      }
      setConfigured(c => ({ ...c, [provider]: false }));
      setSuccess(`${PROVIDERS.find(p => p.key === provider)?.label} API key removed`);
      setTimeout(() => setSuccess(''), 5000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete API key');
    } finally {
      setLoadingProvider(null);
    }
  };

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
                  Add a key for any provider you want to use. All keys are encrypted at rest.
                </p>
              </div>
            </div>
          </div>

          {/* Body */}
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

            <div className="space-y-8">
              {PROVIDERS.map(provider => {
                const isLoading = loadingProvider === provider.key;
                return (
                  <div key={provider.key} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label
                        htmlFor={`apiKey-${provider.key}`}
                        className="block font-mono text-xs text-faint"
                      >
                        {provider.title}
                      </label>
                      {configured[provider.key] && (
                        <span className="inline-flex items-center gap-1 font-mono text-[10px] text-success">
                          <Check className="h-3 w-3" /> Saved
                        </span>
                      )}
                    </div>

                    <p className="font-serif text-xs text-subtle leading-relaxed">{provider.blurb}</p>

                    <div className="flex gap-2">
                      <input
                        type="password"
                        id={`apiKey-${provider.key}`}
                        value={keys[provider.key]}
                        onChange={(e) => setKeys(k => ({ ...k, [provider.key]: e.target.value }))}
                        placeholder={provider.placeholder}
                        className="flex-1 px-3 py-2.5 font-serif text-sm bg-surface border border-hair rounded-sm outline-none transition ease-desk focus:border-accent focus:ring-2 focus:ring-accent/30 placeholder:text-faint placeholder:italic"
                      />
                      <button
                        type="button"
                        onClick={() => handleSave(provider.key)}
                        disabled={isLoading || !keys[provider.key].trim()}
                        className="btn btn-primary disabled:cursor-not-allowed disabled:opacity-60 flex items-center justify-center gap-2 whitespace-nowrap"
                      >
                        {isLoading ? (
                          <span className="inline-block h-4 w-4 animate-spin border-2 border-paper/40 border-t-paper rounded-full" />
                        ) : (
                          <span>Save</span>
                        )}
                      </button>
                      {configured[provider.key] && (
                        <button
                          type="button"
                          onClick={() => handleDelete(provider.key)}
                          disabled={isLoading}
                          className="btn btn-quiet disabled:cursor-not-allowed disabled:opacity-60 whitespace-nowrap"
                          aria-label={`Remove ${provider.label} API key`}
                        >
                          Remove
                        </button>
                      )}
                    </div>

                    <a
                      href={provider.helpUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 mt-1 font-serif text-sm text-accent hover:text-accent-ink transition-colors ease-desk"
                    >
                      {provider.helpText} →
                    </a>
                  </div>
                );
              })}
            </div>

            {/* Security Notice */}
            <div className="mt-8 pt-6 border-t border-hair">
              <div className="flex items-start gap-3 bg-desk border border-hair rounded-sm px-4 py-3">
                <span className="text-accent font-semibold leading-none mt-0.5">⚠</span>
                <div>
                  <p className="font-mono text-xs text-faint mb-1">Security notice</p>
                  <p className="font-serif text-xs text-ink leading-relaxed">
                    Your API keys are encrypted and stored securely. Never share your keys or commit them to version control.
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

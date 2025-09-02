'use client';

import { ArrowLeft } from "lucide-react";
import { useRouter } from "next/navigation";
import React, { useState } from "react";

export default function APISettings() {
  const router = useRouter();
  const [apiKey ,setApiKey] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setSuccess('');

    try {
      const response = await fetch('/api/user/apikey', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({apiKey})
      });
      
      if(!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Faield to save API key');
      }

      setSuccess('API key saved successfully');
      setApiKey('');
    } catch (error) {
      setError(error instanceof Error? error.message : 'Failed to save API key');
    } finally {
      setIsLoading(false);
    }
  }
  return (
  <div className="max-w-md mx-auto">
    <button
      onClick={() => router.back()}
      className="mb-6 flex items-center text-sm text-gray-600 hover:text-gray-900"
    >
      <ArrowLeft size={16} className="mr-1"/>
      Back to Dashboard
    </button>
    <h2 className="text-2xl font-bold mb-4">API Settings</h2>
    <form onSubmit={handleSubmit} className="space-y-4">
      {error && (
        <div className="p-4 text-sm text-red-700 bg-red-100 rounded-lg" role="alert">
          {error}
        </div>
      )}
      {success && (
        <div className="p-4 text-sm text-green-700 bg-green-100 rounded-lg" role="alert">
          {success}
        </div>
      )}
      <div>
        <label htmlFor="apiKey" className="block text-sm font-medium text-gray-700">
          OpenAI API key
        </label>
        <input 
          type="password"
          id="apiKey"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          placeholder="sk-..."
          className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
        />
        <p className="mt-1 text-sm text-gray-500">
          Enter your OpenAI API key to use your own account for chat completion.
        </p>
      </div>
      <button
        type="submit"
        disabled={isLoading || !apiKey}
        className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50">
          {isLoading ? 'Saving...' : 'Save API Key'}
      </button>
    </form>
</div>
 ) 
}
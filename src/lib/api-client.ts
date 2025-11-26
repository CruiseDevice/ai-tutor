// API client that talks to Python backend
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8001';

export async function apiRequest<T = any>(endpoint: string, options?: RequestInit): Promise<Response> {
  const response = await fetch(`${BACKEND_URL}${endpoint}`, {
    ...options,
    credentials: 'include', // Send cookies
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  return response;
}

// Auth API calls
export const authApi = {
  async register(email: string, password: string) {
    return apiRequest('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  },

  async login(email: string, password: string) {
    return apiRequest('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  },

  async logout() {
    return apiRequest('/api/auth/logout', {
      method: 'POST',
    });
  },

  async getUser() {
    return apiRequest('/api/auth/user');
  },

  async verifySession() {
    return apiRequest('/api/auth/verify-session');
  },

  async requestPasswordReset(email: string) {
    return apiRequest('/api/auth/password-reset/request', {
      method: 'POST',
      body: JSON.stringify({ email }),
    });
  },

  async confirmPasswordReset(token: string, new_password: string) {
    return apiRequest('/api/auth/password-reset/confirm', {
      method: 'POST',
      body: JSON.stringify({ token, new_password }),
    });
  },
};

// Document API calls
export const documentApi = {
  async upload(file: File) {
    const formData = new FormData();
    formData.append('file', file);

    return fetch(`${BACKEND_URL}/api/documents`, {
      method: 'POST',
      credentials: 'include',
      body: formData,
      // Don't set Content-Type header - browser will set it with boundary for FormData
    });
  },

  async process(documentId: string) {
    return apiRequest('/api/documents/process', {
      method: 'POST',
      body: JSON.stringify({ document_id: documentId }),
    });
  },

  async list() {
    return apiRequest('/api/documents');
  },

  async delete(documentId: string) {
    return apiRequest(`/api/documents/${documentId}`, {
      method: 'DELETE',
    });
  },

  async get(documentId: string) {
    return apiRequest(`/api/documents/${documentId}`);
  },
};

// Chat API calls
export const chatApi = {
  async sendMessage(conversationId: string, documentId: string, content: string, model?: string) {
    return apiRequest('/api/chat', {
      method: 'POST',
      body: JSON.stringify({
        conversation_id: conversationId,
        document_id: documentId,
        content,
        model: model || 'gpt-4'
      }),
    });
  },

  async sendMessageStream(
    conversationId: string,
    documentId: string,
    content: string,
    model: string,
    onChunk: (chunk: string) => void,
    onDone: (data: any) => void,
    onError: (error: string) => void
  ) {
    const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8001';

    try {
      const response = await fetch(`${BACKEND_URL}/api/chat/stream`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          conversation_id: conversationId,
          document_id: documentId,
          content,
          model: model || 'gpt-4'
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || errorData.error || 'Failed to start streaming');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      if (!reader) {
        throw new Error('No response body reader available');
      }

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === 'chunk') {
                onChunk(data.content);
              } else if (data.type === 'done') {
                onDone(data);
                return;
              } else if (data.type === 'error') {
                onError(data.content);
                return;
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e, line);
            }
          }
        }
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Streaming error occurred';
      onError(errorMessage);
    }
  },
};

// Conversation API calls
export const conversationApi = {
  async list() {
    return apiRequest('/api/conversations');
  },

  async get(conversationId: string) {
    return apiRequest(`/api/conversations/${conversationId}`);
  },

  async delete(conversationId: string) {
    return apiRequest(`/api/conversations/${conversationId}`, {
      method: 'DELETE',
    });
  },
};

// User API calls
export const userApi = {
  async getProfile() {
    return apiRequest('/api/user/profile');
  },

  async updateProfile(email: string) {
    return apiRequest('/api/user/profile', {
      method: 'PUT',
      body: JSON.stringify({ email }),
    });
  },

  async updateAPIKey(apiKey: string) {
    return apiRequest('/api/user/apikey', {
      method: 'POST',
      body: JSON.stringify({ api_key: apiKey }),
    });
  },

  async checkAPIKey() {
    return apiRequest('/api/user/apikey/check');
  },

  async deleteAPIKey() {
    return apiRequest('/api/user/apikey', {
      method: 'DELETE',
    });
  },
};

// Helper function to get JSON from response
export async function getJson<T = any>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: response.statusText }));
    throw new Error(error.error || error.detail || 'Request failed');
  }
  return response.json();
}


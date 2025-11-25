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


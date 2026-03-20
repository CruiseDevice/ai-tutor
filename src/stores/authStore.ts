// src/stores/authStore.ts
import { create } from 'zustand';
import { persist, createJSONStorage, devtools } from 'zustand/middleware';
import { FILE_UPLOAD_LIMITS } from '../config';

interface AuthState {
  // State
  userId: string;
  userEmail: string;
  maxFileSize: number;
  isLoading: boolean;
  isAuthenticated: boolean;

  // Actions
  setUser: (user: { id: string; email: string }) => void;
  setLoading: (loading: boolean) => void;
  setMaxFileSize: (size: number) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  devtools(
    persist(
      (set) => ({
        // Initial State
        userId: '',
        userEmail: '',
        maxFileSize: FILE_UPLOAD_LIMITS.DEFAULT_MAX_BYTES,
        isLoading: false,
        isAuthenticated: false,

        // Actions
        setUser: (user) => set({
          userId: user.id,
          userEmail: user.email,
          isAuthenticated: true,
        }),

        setLoading: (isLoading) => set({ isLoading }),

        setMaxFileSize: (maxFileSize) => set({ maxFileSize }),

        logout: () => set({
          userId: '',
          userEmail: '',
          isAuthenticated: false,
        }),
      }),
      {
        name: 'auth-storage',
        storage: createJSONStorage(() => sessionStorage), // Use sessionStorage, cleared on browser close
        partialize: (state) => ({
          userId: state.userId,
          userEmail: state.userEmail,
          isAuthenticated: state.isAuthenticated,
        }),
      }
    ),
    { name: 'AuthStore' }
  )
);

// Selectors for optimized re-renders
export const selectUser = (state: AuthState) => ({
  userId: state.userId,
  userEmail: state.userEmail,
});

export const selectIsAuthenticated = (state: AuthState) => state.isAuthenticated;
export const selectIsLoading = (state: AuthState) => state.isLoading;
export const selectMaxFileSize = (state: AuthState) => state.maxFileSize;

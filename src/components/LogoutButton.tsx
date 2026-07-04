// app/components/LogoutButton.tsx
'use client';

import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { authApi } from '@/lib/api-client';

export default function LogoutButton() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);

  const handleLogout = async () => {
    setIsLoading(true);
    try {
      const response = await authApi.logout();

      if (!response.ok) {
        throw new Error('Logout failed');
      }

      // Redirect to login page after successful logout
      router.push('/login');
      // Force a page refresh to clear any client-side state
      router.refresh();
    } catch (error) {
      // TODO: Display error message to user
      console.error('Logout error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <button
      onClick={handleLogout}
      disabled={isLoading}
      className="btn btn-quiet disabled:opacity-50"
    >
      {isLoading ? 'Logging out…' : 'Log out'}
    </button>
  );
}
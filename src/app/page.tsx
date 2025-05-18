// app/page.tsx
"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function Home() {
  const router = useRouter();
  useEffect(() => {
    // check if user is authenticated
    const checkAuth = async () => {
      try {
        const response = await fetch('/api/auth/verify-session');

        // if authenticated, redirect to dashboard
        if(response.ok) {
          router.push('/dashboard')
        } else {
          // if not authenticated, redirect to login
          router.push('/login');
        }
      } catch (error) {
        // on error, redirect to login
        router.push('/login');
        console.error('Auth check error: ', error);
      }
    };
    checkAuth();
  }, [router]);

  return (
    <div className="h-screen flex items-center justify-center bg-gray-50">
      <div className="text-center">
        <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-soid border-blue-500 border-r-transparent"></div>
          <p className="mt-4 text-gray-600">Redirecting...</p>
      </div>
    </div>
  );
}

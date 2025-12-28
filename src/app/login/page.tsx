// app/login/page.tsx
"use client";

import LoginForm from "@/components/LoginForm";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { authApi } from "@/lib/api-client";

export default function LoginPage() {
  const router = useRouter();
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    // Check if user is already authenticated
    const checkAuth = async () => {
      try {
        const response = await authApi.verifySession();

        // If already authenticated, redirect to dashboard
        if (response.ok) {
          router.push('/dashboard');
        } else {
          // Not authenticated, show login form
          setIsChecking(false);
        }
      } catch (error) {
        // On error, show login form
        setIsChecking(false);
        console.error('Auth check error:', error);
      }
    };

    checkAuth();
  }, [router]);

  // Show loading state while checking authentication
  if (isChecking) {
    return (
      <div className="h-screen flex items-center justify-center bg-gradient-to-br from-purple-50 via-blue-50 to-pink-50">
        <div className="text-center">
          <div className="relative inline-block">
            <div className="absolute inset-0 rounded-full bg-gradient-to-r from-purple-600 to-blue-600 opacity-20 animate-pulse-glow"></div>
            <div className="relative inline-block h-12 w-12 animate-spin rounded-full border-4 border-purple-200 border-t-purple-600 border-r-blue-600"></div>
          </div>
          <p className="mt-6 text-gray-700 font-medium">Checking authentication...</p>
        </div>
      </div>
    );
  }

  return <LoginForm />;
}
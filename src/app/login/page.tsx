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
      <div className="h-screen flex items-center justify-center bg-paper">
        <div className="text-center">
          <div className="inline-block h-12 w-12 border-4 border-ink border-t-accent animate-spin rounded-full"></div>
          <p className="mt-6 font-mono text-sm text-ink">[Checking authentication...]</p>
        </div>
      </div>
    );
  }

  return <LoginForm />;
}
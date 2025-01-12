'use client';

import Link from 'next/link';
import LogoutButton from './LogoutButton';
import { useEffect, useState } from 'react';
import { usePathname } from 'next/navigation';

export default function Navbar() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const response = await fetch('/api/auth/verify-session');
        setIsAuthenticated(response.ok);
      } catch (error) {
        console.error('Auth check error: ', error);
        setIsAuthenticated(false);
      }
    };
    checkAuth();
  }, [pathname]);

  return (
    <nav className="bg-white shadow">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link href="/dashboard" className="text-lg font-semibold">
              StudyFetch
            </Link>
          </div>
          {isAuthenticated ? (
            <div className="flex items-center">
              <Link 
                href="/settings" 
                className="text-sm font-medium text-gray-700 hover:text-gray-900 px-4 py-2 rounded-md hover:bg-gray-100"
              >
                API Settings
              </Link>
              <LogoutButton />
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <Link 
                href="/login"
                className="text-sm font-medium text-gray-700 hover:text-gray-900 px-4 py-2 rounded-md hover:bg-gray-100"
              >
                Login
              </Link>
              <Link 
                href="/register"
                className="text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 px-4 py-2 rounded-md"
              >
                Sign up
              </Link>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
}
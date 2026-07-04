"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { authApi } from '@/lib/api-client';
import { Activity, Users, Settings } from 'lucide-react';

export default function AdminLayout({
  children
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [isAdmin, setIsAdmin] = useState(false);

  // Check admin access
  useEffect(() => {
    const checkAdminAccess = async () => {
      try {
        const res = await authApi.getUser();
        if (!res.ok) {
          router.push('/login');
          return;
        }

        const user = await res.json();

        // Check if user has admin role
        if (user.role !== 'admin' && user.role !== 'super_admin') {
          router.push('/');
          return;
        }

        setIsAdmin(true);
      } catch (err) {
        console.error('Failed to verify admin access:', err);
        router.push('/');
      } finally {
        setLoading(false);
      }
    };

    checkAdminAccess();
  }, [router]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-desk">
        <div className="animate-spin w-8 h-8 border-2 border-hair border-t-accent rounded-full" />
      </div>
    );
  }

  if (!isAdmin) {
    return null;
  }

  return (
    <div className="min-h-screen bg-desk">
      {/* Admin Navigation */}
      <nav className="bg-paper border-b border-hair">
        <div className="container mx-auto px-4">
          <div className="flex items-center gap-6 h-16">
            <h1 className="font-serif text-xl font-semibold text-ink">Admin Panel</h1>

            <div className="flex gap-2">
              <a
                href="/admin/queue"
                className="flex items-center gap-2 px-3 py-2 rounded-sm text-subtle hover:text-accent hover:bg-accent-soft/50 transition-colors font-serif text-sm"
              >
                <Activity className="w-4 h-4" />
                Queue Monitor
              </a>
              <a
                href="/admin/users"
                className="flex items-center gap-2 px-3 py-2 rounded-sm text-faint cursor-not-allowed font-serif text-sm"
              >
                <Users className="w-4 h-4" />
                Users
              </a>
              <a
                href="/admin/settings"
                className="flex items-center gap-2 px-3 py-2 rounded-sm text-faint cursor-not-allowed font-serif text-sm"
              >
                <Settings className="w-4 h-4" />
                Settings
              </a>
            </div>
          </div>
        </div>
      </nav>

      {/* Content */}
      <main>{children}</main>
    </div>
  );
}

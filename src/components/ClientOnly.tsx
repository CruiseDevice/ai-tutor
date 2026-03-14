'use client';

import { useEffect, useState } from 'react';

/**
 * Wraps children to render only on client side.
 * Prevents hydration errors when code accesses browser APIs (localStorage, sessionStorage).
 */
export function ClientOnly({ children }: { children: React.ReactNode }) {
  const [hasMounted, setHasMounted] = useState(false);

  useEffect(() => {
    setHasMounted(true);
  }, []);

  if (!hasMounted) {
    return null;
  }

  return <>{children}</>;
}

import { useRef, useCallback, useEffect } from 'react';

/**
 * Throttles a callback function, ensuring it's not called more frequently than the specified delay.
 * Unlike debounce, throttling guarantees the function runs at regular intervals.
 *
 * @param callback - The function to throttle
 * @param delay - Minimum time between callback invocations in milliseconds
 * @returns A throttled version of the callback
 */
export function useThrottledCallback<T extends (...args: any[]) => any>(
  callback: T,
  delay: number
): T {
  const lastRunRef = useRef<number>(0);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  return useCallback(
    (...args: Parameters<T>) => {
      const now = Date.now();
      const timeSinceLastRun = now - lastRunRef.current;

      const runCallback = () => {
        lastRunRef.current = Date.now();
        callback(...args);
      };

      if (timeSinceLastRun >= delay) {
        // Enough time has passed, run immediately
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
          timeoutRef.current = null;
        }
        runCallback();
      } else {
        // Not enough time, schedule to run at next available slot
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
        }
        const timeUntilNextRun = delay - timeSinceLastRun;
        timeoutRef.current = setTimeout(runCallback, timeUntilNextRun);
      }
    },
    [callback, delay]
  ) as T;

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);
}

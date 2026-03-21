import { useState, useCallback, RefObject } from 'react';

export interface UsePDFNavigationOptions {
  numPages: number | null;
  onPageChange?: (page: number) => void;
  containerRef?: RefObject<HTMLDivElement>;
}

export interface UsePDFNavigationReturn {
  pageNumber: number;
  goToPage: (page: number) => void;
  nextPage: () => void;
  prevPage: () => void;
  canGoNext: boolean;
  canGoPrev: boolean;
  touchHandlers: {
    onTouchStart: (e: React.TouchEvent) => void;
    onTouchMove: (e: React.TouchEvent) => void;
    onTouchEnd: () => void;
  };
}

const MIN_SWIPE_DISTANCE = 50;

/**
 * Hook for managing PDF page navigation with touch gesture support.
 *
 * @example
 * const { pageNumber, goToPage, nextPage, prevPage, canGoNext, canGoPrev, touchHandlers } =
 *   usePDFNavigation({ numPages, onPageChange: handlePageChange });
 */
export function usePDFNavigation({
  numPages,
  onPageChange,
}: UsePDFNavigationOptions): UsePDFNavigationReturn {
  const [pageNumber, setPageNumber] = useState(1);
  const [touchStart, setTouchStart] = useState<number | null>(null);
  const [touchEnd, setTouchEnd] = useState<number | null>(null);

  const goToPage = useCallback((page: number) => {
    const maxPages = numPages || 1;
    if (page >= 1 && page <= maxPages) {
      setPageNumber(page);
      onPageChange?.(page);
    }
  }, [numPages, onPageChange]);

  const nextPage = useCallback(() => {
    goToPage(pageNumber + 1);
  }, [pageNumber, goToPage]);

  const prevPage = useCallback(() => {
    goToPage(pageNumber - 1);
  }, [pageNumber, goToPage]);

  const canGoNext = pageNumber < (numPages || 1);
  const canGoPrev = pageNumber > 1;

  // Touch swipe gesture handlers for page navigation
  const onTouchStart = useCallback((e: React.TouchEvent) => {
    setTouchStart(e.touches[0].clientX);
    setTouchEnd(null);
  }, []);

  const onTouchMove = useCallback((e: React.TouchEvent) => {
    setTouchEnd(e.touches[0].clientX);
  }, []);

  const onTouchEnd = useCallback(() => {
    if (touchStart === null || touchEnd === null) return;

    const distance = touchStart - touchEnd;
    const isLeftSwipe = distance > MIN_SWIPE_DISTANCE;
    const isRightSwipe = distance < -MIN_SWIPE_DISTANCE;

    if (isLeftSwipe && canGoNext) {
      // Swiped left → next page
      nextPage();
    } else if (isRightSwipe && canGoPrev) {
      // Swiped right → previous page
      prevPage();
    }

    // Reset for next gesture
    setTouchStart(null);
    setTouchEnd(null);
  }, [touchStart, touchEnd, canGoNext, canGoPrev, nextPage, prevPage]);

  return {
    pageNumber,
    goToPage,
    nextPage,
    prevPage,
    canGoNext,
    canGoPrev,
    touchHandlers: {
      onTouchStart,
      onTouchMove,
      onTouchEnd,
    },
  };
}

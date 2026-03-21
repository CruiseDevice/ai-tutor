// app/components/EnhancedPDFViewer.tsx
import { Loader2, Upload, Eye, EyeOff } from "lucide-react";
import { useRef, useState, useCallback, useImperativeHandle, forwardRef, useEffect, useMemo } from "react";
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/TextLayer.css';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import type { PDFAnnotation, AnnotationReference } from '@/types/annotations';
import { useThrottledCallback } from '@/hooks/useThrottledCallback';
import { usePDFNavigation } from '@/hooks/usePDFNavigation';
import { loadPDFState, savePDFState } from '@/utils/pdfStatePersistence';

// Store imports for Zustand migration
import { useChatStore } from '@/stores/chatStore';
import { useAnnotationsStore, selectAnnotations } from '@/stores/annotationsStore';

// Initialize pdfjs worker
// Use explicit HTTPS to avoid Safari iOS CORS issues with protocol-relative URLs
pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.mjs`;

// Annotation Shape Component
interface AnnotationShapeProps {
  annotation: PDFAnnotation;
  onClick?: () => void;
}

function AnnotationShape({ annotation, onClick }: AnnotationShapeProps) {
  const { type, bounds, color } = annotation;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onClick?.();
    }
  };

  const baseStyle: React.CSSProperties = {
    position: 'absolute',
    left: `${bounds.x}%`,
    top: `${bounds.y}%`,
    width: `${bounds.width}%`,
    height: `${bounds.height}%`,
    pointerEvents: 'auto',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
  };

  const ariaLabel = `${type} annotation${annotation.textContent ? `: ${annotation.textContent}` : ''}`;

  switch (type) {
    case 'highlight':
      return (
        <div
          role="button"
          tabIndex={0}
          aria-label={ariaLabel}
          style={{
            ...baseStyle,
            backgroundColor: color || 'rgba(212, 82, 0, 0.4)',
          }}
          onClick={onClick}
          onKeyDown={handleKeyDown}
          className="hover:brightness-110"
        />
      );

    case 'circle':
      return (
        <div
          role="button"
          tabIndex={0}
          aria-label={ariaLabel}
          style={{
            ...baseStyle,
            border: `3px solid ${color || 'rgba(10, 10, 10, 0.8)'}`,
            borderRadius: '50%',
            backgroundColor: 'transparent',
          }}
          onClick={onClick}
          onKeyDown={handleKeyDown}
          className="animate-pulse"
        />
      );

    case 'box':
      return (
        <div
          role="button"
          tabIndex={0}
          aria-label={ariaLabel}
          style={{
            ...baseStyle,
            border: `3px solid ${color || 'rgba(10, 10, 10, 0.8)'}`,
            backgroundColor: color?.replace('0.8', '0.1') || 'rgba(10, 10, 10, 0.1)',
          }}
          onClick={onClick}
          onKeyDown={handleKeyDown}
        />
      );

    case 'underline':
      return (
        <div
          role="button"
          tabIndex={0}
          aria-label={ariaLabel}
          style={{
            ...baseStyle,
            height: '3px',
            top: `${bounds.y + bounds.height}%`,
            backgroundColor: color || 'rgba(212, 82, 0, 0.8)',
          }}
          onClick={onClick}
          onKeyDown={handleKeyDown}
        />
      );

    default:
      return null;
  }
}

export interface PDFViewerRef {
  goToPage: (pageNum: number) => void;
  setAnnotations: (annotations: AnnotationReference[]) => void;
  clearAnnotations: () => void;
  highlightText: (pageNum: number, textToFind: string) => void;
}

interface EnhancedPDFViewerProps {
  // Required handlers
  onFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;

  // Ref for file input (from parent)
  fileInputRef: React.RefObject<HTMLInputElement>;

  // Optional collapse handler
  onCollapse?: () => void;
}

const EnhancedPDFViewer = forwardRef<PDFViewerRef, EnhancedPDFViewerProps>(({
  onFileUpload,
  fileInputRef: externalFileInputRef,
  onCollapse,
}, ref) => {

  const internalFileInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = externalFileInputRef || internalFileInputRef;
  const pageInputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const pageContainerRef = useRef<HTMLDivElement>(null);

  // Load saved PDF state on mount
  const savedState = useMemo(() => loadPDFState(), []);

  const [numPages, setNumPages] = useState<number | null>(null);
  const [scale, setScale] = useState(() => savedState?.scale ?? 1.0);
  const [rotation, setRotation] = useState(() => savedState?.rotation ?? 0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [pageInputError, setPageInputError] = useState<string | null>(null);
  const [localAnnotations, setLocalAnnotations] = useState<AnnotationReference[]>([]);
  const [showAnnotations, setShowAnnotations] = useState(true);
  const [highlightedTextRects, setHighlightedTextRects] = useState<{x: number, y: number, width: number, height: number}[]>([]);

  // PDF data state - fetch PDF with credentials to avoid 401 errors
  // Use Blob URL instead of Uint8Array to avoid DataCloneError with large PDFs
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [loadingPdf, setLoadingPdf] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);

  // =====================================================
  // STORE HOOKS - PDF data now managed by Zustand stores
  // =====================================================
  const currentPDF = useChatStore((s) => s.currentPDF);
  const storeAnnotations = useAnnotationsStore(selectAnnotations);
  const storeSetSelectedAnnotation = useAnnotationsStore((s) => s.setSelectedAnnotation);

  // Combine external (store) and local annotations
  const allAnnotations = useMemo(() => [...storeAnnotations, ...localAnnotations], [storeAnnotations, localAnnotations]);

  // =====================================================
  // NAVIGATION HOOK - Manages page state and touch gestures
  // =====================================================
  const {
    pageNumber,
    goToPage,
    nextPage,
    prevPage,
    canGoNext,
    canGoPrev,
    touchHandlers,
  } = usePDFNavigation({
    numPages,
    onPageChange: (page) => setPageInputError(null),
  });

  // Get annotations for current page
  const currentPageAnnotations = useMemo(
    () => allAnnotations.filter(a => a.pageNumber === pageNumber),
    [allAnnotations, pageNumber]
  );

  // Persist zoom/rotation state to localStorage
  useEffect(() => {
    savePDFState({ scale, rotation });
  }, [scale, rotation]);

  // Handler for annotation click - delegates to store
  const handleAnnotationClick = useCallback((annotationRef: AnnotationReference) => {
    storeSetSelectedAnnotation(annotationRef.pageNumber.toString());
  }, [storeSetSelectedAnnotation]);

  // Function to find text on the current page and get its position
  const findTextOnPage = useCallback((searchText: string) => {
    if (!pageContainerRef.current || !searchText) {
      console.log('[PDF Annotation] No container or search text');
      return;
    }

    const textLayer = pageContainerRef.current.querySelector('.react-pdf__Page__textContent');
    if (!textLayer) {
      console.log('[PDF Annotation] Text layer not found, retrying...');
      // Retry after a short delay if text layer isn't ready
      setTimeout(() => findTextOnPage(searchText), 300);
      return;
    }

    const textSpans = textLayer.querySelectorAll('span');
    const searchLower = searchText.toLowerCase().trim();
    const searchWords = searchLower.split(/\s+/).filter(w => w.length > 2);
    const rects: {x: number, y: number, width: number, height: number}[] = [];

    console.log(`[PDF Annotation] Searching for: "${searchText}" (${textSpans.length} spans on page)`);

    // Strategy 1: Look for spans containing significant words from search text
    const matchingSpans: Element[] = [];

    textSpans.forEach((span) => {
      const spanText = span.textContent?.toLowerCase() || '';
      if (!spanText.trim()) return;

      // Check if span contains any of the significant search words
      const matchesWord = searchWords.some(word => spanText.includes(word));
      // Or if search text contains the span text (for short spans)
      const spanContainedInSearch = spanText.trim().length > 3 && searchLower.includes(spanText.trim());

      if (matchesWord || spanContainedInSearch) {
        matchingSpans.push(span);
      }
    });

    console.log(`[PDF Annotation] Found ${matchingSpans.length} matching spans`);

    // Get positions of matching spans
    matchingSpans.forEach((span) => {
      const rect = span.getBoundingClientRect();
      const containerRect = pageContainerRef.current!.getBoundingClientRect();

      if (rect.width > 0 && rect.height > 0) {
        rects.push({
          x: rect.left - containerRect.left,
          y: rect.top - containerRect.top,
          width: rect.width,
          height: rect.height
        });
      }
    });

    // Strategy 2: If no matches, try fuzzy word matching
    if (rects.length === 0 && searchWords.length > 0) {
      console.log('[PDF Annotation] Trying fuzzy match with first word:', searchWords[0]);
      textSpans.forEach((span) => {
        const spanText = span.textContent?.toLowerCase() || '';
        if (spanText.includes(searchWords[0])) {
          const rect = span.getBoundingClientRect();
          const containerRect = pageContainerRef.current!.getBoundingClientRect();

          if (rect.width > 0 && rect.height > 0) {
            rects.push({
              x: rect.left - containerRect.left,
              y: rect.top - containerRect.top,
              width: rect.width,
              height: rect.height
            });
          }
        }
      });
    }

    console.log(`[PDF Annotation] Final highlight rects: ${rects.length}`);
    setHighlightedTextRects(rects);
  }, []);

  // Throttled version to avoid excessive searches during rapid changes
  const throttledFindText = useThrottledCallback(findTextOnPage, 300);

  // Expose methods to parent via ref
  useImperativeHandle(ref, () => ({
    goToPage: (pageNum: number) => {
      console.log(`[PDF Annotation] goToPage called: ${pageNum}`);
      goToPage(pageNum);
    },
    setAnnotations: (annotations: AnnotationReference[]) => {
      console.log(`[PDF Annotation] setAnnotations called:`, annotations);
      setLocalAnnotations(annotations);
    },
    clearAnnotations: () => {
      console.log('[PDF Annotation] clearAnnotations called');
      setLocalAnnotations([]);
      setHighlightedTextRects([]);
    },
    highlightText: (pageNum: number, textToFind: string) => {
      console.log(`[PDF Annotation] highlightText called: page ${pageNum}, text "${textToFind}"`);
      goToPage(pageNum);
      // Text highlighting will be handled by throttledFindText after page renders
      setTimeout(() => throttledFindText(textToFind), 500);
    }
  }), [goToPage, throttledFindText]);

  // Effect to find text when annotations change
  useEffect(() => {
    if (currentPageAnnotations.length > 0 && showAnnotations) {
      const textTargets = new Set<string>();
      currentPageAnnotations.forEach(annotationRef => {
        annotationRef.annotations.forEach(annotation => {
          if (annotation.textContent) {
            textTargets.add(annotation.textContent);
          }
        });
        if (annotationRef.sourceText) {
          textTargets.add(annotationRef.sourceText);
        }
      });
      // Use throttled version to avoid excessive searches
      textTargets.forEach(text => throttledFindText(text));
    } else {
      setHighlightedTextRects([]);
    }
  }, [currentPageAnnotations, showAnnotations, throttledFindText, pageNumber]);

  // Fetch PDF with credentials when currentPDF changes
  // This avoids 401 errors since PDF.js worker doesn't include credentials
  useEffect(() => {
    let objectUrl: string | null = null;

    const fetchPdfWithCredentials = async () => {
      // Revoke previous URL if exists
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
        objectUrl = null;
      }

      if (!currentPDF) {
        setPdfUrl(null);
        setLoadingPdf(false);
        setPdfError(null);
        return;
      }

      setLoadingPdf(true);
      setPdfError(null);

      try {
        console.log('[PDF Viewer] Fetching PDF with credentials:', currentPDF);
        const response = await fetch(currentPDF, {
          credentials: 'include',
          cache: 'force-cache', // Cache the PDF for better performance
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch PDF: ${response.status} ${response.statusText}`);
        }

        const blob = await response.blob();
        objectUrl = URL.createObjectURL(blob);

        console.log('[PDF Viewer] PDF loaded successfully, size:', blob.size);
        setPdfUrl(objectUrl);
      } catch (err) {
        console.error('[PDF Viewer] Error loading PDF:', err);
        setPdfError(err instanceof Error ? err.message : 'Failed to load PDF');
        setPdfUrl(null);
      } finally {
        setLoadingPdf(false);
      }
    };

    fetchPdfWithCredentials();

    // Cleanup: revoke object URL when effect runs again or component unmounts
    return () => {
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [currentPDF]);

  // Handler for page load success - triggers text search for current annotations
  const handlePageLoadSuccess = useCallback(() => {
    console.log('[PDF Annotation] Page rendered successfully');
    // Re-trigger text search for current annotations after page renders
    if (currentPageAnnotations.length > 0 && showAnnotations) {
      const textTargets = new Set<string>();
      currentPageAnnotations.forEach(annotationRef => {
        annotationRef.annotations.forEach(annotation => {
          if (annotation.textContent) {
            textTargets.add(annotation.textContent);
          }
        });
        if (annotationRef.sourceText) {
          textTargets.add(annotationRef.sourceText);
        }
      });
      // Small delay for text layer to be ready, then use throttled search
      textTargets.forEach(text => {
        setTimeout(() => throttledFindText(text), 100);
      });
    }
  }, [currentPageAnnotations, showAnnotations, throttledFindText]);

  const handlePageInputChange = () => {
    if (pageInputError) setPageInputError(null);
  }

  const handlePageInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handlePageJump();
    }
  }

  const handlePageJump = () => {
    const inputValue = pageInputRef.current?.value;
    if (!inputValue) return;

    const pageNum = parseInt(inputValue);

    if (isNaN(pageNum)) {
      setPageInputError('Invalid page');
      return;
    }

    if (pageNum < 1 || pageNum > (numPages || 1)) {
      setPageInputError(`1 - ${numPages || 1}`);
      return;
    }

    goToPage(pageNum);
    if (pageInputRef.current) {
      pageInputRef.current.value = pageNum.toString();
      pageInputRef.current.blur();
    }
    // Restore focus to PDF container for keyboard users
    // Small delay to allow page to render
    setTimeout(() => pageContainerRef.current?.focus(), 100);
  }

  // Keyboard navigation for PDF container
  const handleContainerKeyDown = (e: React.KeyboardEvent) => {
    // Only handle if not in an input
    if ((e.target as HTMLElement).tagName === 'INPUT') return;

    switch (e.key) {
      case 'ArrowLeft':
        if (pageNumber > 1) {
          e.preventDefault();
          goToPage(pageNumber - 1);
        }
        break;
      case 'ArrowRight':
        if (pageNumber < (numPages || 1)) {
          e.preventDefault();
          goToPage(pageNumber + 1);
        }
        break;
      case 'Home':
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          goToPage(1);
        }
        break;
      case 'End':
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          goToPage(numPages || 1);
        }
        break;
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsProcessing(true);
    try {
      await onFileUpload(e);
    } finally {
      setIsProcessing(false);
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type === 'application/pdf') {
      // Create a synthetic event to reuse the existing handler
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(files[0]);

      const syntheticEvent = {
        target: { files: dataTransfer.files }
      } as unknown as React.ChangeEvent<HTMLInputElement>;

      handleFileUpload(syntheticEvent);
    }
  };

  const onDocumentLoadSuccess = ({numPages}: {numPages: number}) => {
    setNumPages(numPages);
    goToPage(1);
  }

  const rotate = () => {
    setRotation((prev) => (prev + 90) % 360);
  };

  return (
    <div className="w-full h-full flex flex-col bg-panel-bg relative overflow-hidden">
      {/* =====================================================
          [003] HEADER - Document Title & Controls
          ===================================================== */}
      {currentPDF && (
        <div className="border-b-2 border-ink bg-panel-bg">
          <div className="flex items-center justify-between px-4 py-3">
            {/* Left: Panel Number & Title */}
            <div className="flex items-center gap-3 overflow-hidden">
              <span className="font-mono text-xs text-accent">[003]</span>
              <h3 className="font-mono text-sm font-bold truncate max-w-md">
                {currentPDF.split('/').pop() || 'Document'}
              </h3>
            </div>

            {/* Right: Page Info & Actions */}
            <div className="flex items-center gap-3">
              {/* Live region for screen readers - announces annotation changes */}
              <div
                role="status"
                aria-live="polite"
                className="sr-only"
              >
                {currentPageAnnotations.length > 0
                  ? `${currentPageAnnotations.length} annotation${currentPageAnnotations.length > 1 ? 's' : ''} on this page`
                  : 'No annotations on this page'
                }
              </div>

              <span className="font-mono text-xs text-subtle">
                p.{pageNumber}/{numPages || '-'}
              </span>

              {/* Annotation Toggle */}
              {allAnnotations.length > 0 && (
                <button
                  onClick={() => setShowAnnotations(!showAnnotations)}
                  className={`no-select font-mono text-xs px-2 py-1 border transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center ${
                    showAnnotations
                      ? 'bg-accent text-paper border-accent'
                      : 'border-ink hover:bg-ink hover:text-paper'
                  }`}
                  title={showAnnotations ? 'Hide Annotations' : 'Show Annotations'}
                >
                  {showAnnotations ? '[§ ON]' : '[§ OFF]'}
                </button>
              )}

              {/* Collapse Button - Only show when onCollapse is provided */}
              {onCollapse && (
                <button
                  onClick={onCollapse}
                  className="no-select font-mono text-xs px-2 py-1 border border-ink hover:bg-ink hover:text-paper transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
                  title="Hide PDF Viewer"
                >
                  [_]
                </button>
              )}

              <button
                onClick={rotate}
                className="no-select font-mono text-xs px-2 py-1 border border-ink hover:bg-ink hover:text-paper transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
                title="Rotate Page"
              >
                [↻]
              </button>
            </div>
          </div>

          {/* Control Bar - Brutalist Style */}
          <div className="flex items-center justify-center gap-4 px-4 py-2 border-t border-ink">
            {/* Zoom Controls */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setScale(prev => Math.max(0.5, prev - 0.1))}
                className="no-select font-mono text-xs px-2 py-1 border border-ink hover:bg-ink hover:text-paper transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
                title="Zoom Out"
              >
                [−]
              </button>
              <span className="font-mono text-xs w-12 text-center">
                {Math.round(scale * 100)}%
              </span>
              <button
                onClick={() => setScale(prev => Math.min(2, prev + 0.1))}
                className="no-select font-mono text-xs px-2 py-1 border border-ink hover:bg-ink hover:text-paper transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
                title="Zoom In"
              >
                [+]
              </button>
            </div>

            {/* Divider */}
            <div className="w-px h-6 bg-ink"></div>

            {/* Page Navigation */}
            <div className="flex items-center gap-2">
              <button
                onClick={prevPage}
                disabled={!canGoPrev}
                className="no-select font-mono text-xs px-2 py-1 border border-ink hover:bg-ink hover:text-paper disabled:opacity-30 disabled:hover:bg-transparent transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
              >
                [◀]
              </button>

              <div className="flex items-center gap-1 font-mono text-xs bg-paper border border-ink px-2 py-1">
                <input
                  ref={pageInputRef}
                  type="text"
                  defaultValue={pageNumber}
                  key={pageNumber}
                  className="w-8 text-center bg-transparent focus:outline-none font-mono"
                  onChange={handlePageInputChange}
                  onKeyDown={handlePageInputKeyDown}
                  onFocus={(e) => e.target.select()}
                  aria-label="Go to page"
                />
                <span className="text-subtle">/ {numPages || '-'}</span>
              </div>

              <button
                onClick={nextPage}
                disabled={!canGoNext}
                className="no-select font-mono text-xs px-2 py-1 border border-ink hover:bg-ink hover:text-paper disabled:opacity-30 disabled:hover:bg-transparent transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
              >
                [▶]
              </button>
            </div>

            {/* Error Toast */}
            {pageInputError && (
              <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 px-3 py-1 bg-accent text-paper font-mono text-xs border border-accent">
                [{pageInputError}]
              </div>
            )}
          </div>
        </div>
      )}

      {/* =====================================================
          CONTENT AREA
          ===================================================== */}
      <div
        className="flex-1 overflow-auto relative z-10 scrollbar-thin scrollbar-thumb-slate-200"
        ref={containerRef}
      >
        {loadingPdf ? (
          <div className="flex flex-col items-center justify-center h-full gap-6 p-8">
            {/* PDF skeleton matching brutalist aesthetic */}
            <div className="w-full max-w-md space-y-3">
              {/* Document page skeleton */}
              <div className="h-80 bg-paper border-2 border-ink/30 relative overflow-hidden">
                {/* Skeleton content lines */}
                <div className="absolute top-8 left-8 right-8 space-y-2">
                  <div className="h-3 bg-subtle/20 animate-pulse w-3/4" />
                  <div className="h-3 bg-subtle/20 animate-pulse delay-75 w-full" />
                  <div className="h-3 bg-subtle/20 animate-pulse delay-100 w-5/6" />
                </div>
                <div className="absolute top-20 left-8 right-8 space-y-2">
                  <div className="h-3 bg-subtle/20 animate-pulse w-full" />
                  <div className="h-3 bg-subtle/20 animate-pulse delay-75 w-2/3" />
                </div>
                <div className="absolute top-32 left-8 right-8 space-y-2">
                  <div className="h-3 bg-subtle/20 animate-pulse w-4/5" />
                  <div className="h-3 bg-subtle/20 animate-pulse delay-75 w-full" />
                  <div className="h-3 bg-subtle/20 animate-pulse delay-100 w-3/4" />
                </div>
              </div>
              {/* Skeleton page indicator */}
              <div className="flex justify-center">
                <div className="h-6 w-24 bg-subtle/20 animate-pulse border border-ink/30" />
              </div>
            </div>
            <span className="font-mono text-sm text-subtle">[LOADING DOCUMENT...]</span>
          </div>
        ) : pdfError ? (
          <div className="flex flex-col items-center justify-center h-64 gap-3 text-accent">
            <span className="font-mono text-4xl">[!]</span>
            <span className="font-serif font-medium">Failed to load PDF</span>
            <span className="font-mono text-xs text-subtle">{pdfError}</span>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="brutalist-button brutalist-button-primary font-mono text-xs px-4 py-2 mt-2 min-h-[44px]"
            >
              [TRY AGAIN]
            </button>
          </div>
        ) : pdfUrl ? (
          <div className="flex justify-center min-h-full p-4">
            <Document
              file={pdfUrl}
              onLoadSuccess={onDocumentLoadSuccess}
              loading={
                <div className="flex flex-col items-center justify-center h-64 gap-4">
                  {/* Page skeleton */}
                  <div className="w-64 space-y-2">
                    <div className="h-48 bg-paper border-2 border-ink/30 p-4 space-y-2">
                      <div className="h-2 bg-subtle/20 animate-pulse w-full" />
                      <div className="h-2 bg-subtle/20 animate-pulse delay-75 w-4/5" />
                      <div className="h-2 bg-subtle/20 animate-pulse delay-100 w-11/12" />
                    </div>
                    <div className="h-4 bg-subtle/20 animate-pulse w-20 mx-auto" />
                  </div>
                  <span className="font-mono text-sm text-subtle">[LOADING PAGE...]</span>
                </div>
              }
              error={
                <div className="flex flex-col items-center justify-center h-64 gap-2 text-accent">
                  <span className="font-mono text-4xl">[!]</span>
                  <span className="font-serif font-medium">Failed to load PDF</span>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="brutalist-button brutalist-button-primary font-mono text-xs px-4 py-2 mt-2 min-h-[44px]"
                  >
                    [TRY AGAIN]
                  </button>
                </div>
              }
              className="pdf-document"
            >
              <div
                ref={pageContainerRef}
                tabIndex={showAnnotations ? 0 : -1}
                aria-label={`PDF page ${pageNumber} of ${numPages || '?'}${currentPageAnnotations.length > 0 ? ` with ${currentPageAnnotations.length} annotation${currentPageAnnotations.length > 1 ? 's' : ''}` : ''}`}
                onTouchStart={touchHandlers.onTouchStart}
                onTouchMove={touchHandlers.onTouchMove}
                onTouchEnd={touchHandlers.onTouchEnd}
                onKeyDown={handleContainerKeyDown}
                className="relative border-2 border-ink bg-white focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2"
                style={{
                  transformOrigin: 'top center'
                }}
              >
                <Page
                  pageNumber={pageNumber}
                  scale={scale}
                  rotate={rotation}
                  renderAnnotationLayer={true}
                  renderTextLayer={true}
                  className="bg-white"
                  width={containerRef.current?.clientWidth ? Math.min(containerRef.current.clientWidth - 48, 800) : undefined}
                  onRenderSuccess={handlePageLoadSuccess}
                />

                {/* Annotation Overlay Layer */}
                {showAnnotations && (
                  <div className="absolute inset-0 pointer-events-none z-10">
                    {/* Render highlight rectangles from text search */}
                    {highlightedTextRects.map((rect, idx) => {
                      const firstAnnotation = currentPageAnnotations[0];
                      const ariaLabel = firstAnnotation?.sourceText
                        ? `Annotation: ${firstAnnotation.sourceText}`
                        : 'View annotation details';

                      return (
                        <div
                          key={`highlight-rect-${idx}`}
                          role="button"
                          tabIndex={0}
                          aria-label={ariaLabel}
                          className="absolute pointer-events-auto cursor-pointer transition-all duration-200 animate-pulse"
                          style={{
                            left: rect.x,
                            top: rect.y,
                            width: rect.width,
                            height: rect.height,
                            backgroundColor: 'rgba(212, 82, 0, 0.4)',
                            border: '2px solid rgba(212, 82, 0, 0.8)',
                          }}
                          onClick={() => {
                            if (firstAnnotation) {
                              handleAnnotationClick(firstAnnotation);
                            }
                          }}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault();
                              if (firstAnnotation) {
                                handleAnnotationClick(firstAnnotation);
                              }
                            }
                          }}
                        />
                      );
                    })}

                    {/* Render annotation shapes from annotation data */}
                    {currentPageAnnotations.map((annotationRef) =>
                      annotationRef.annotations.map((annotation) => (
                        <AnnotationShape
                          key={annotation.id}
                          annotation={annotation}
                          onClick={() => handleAnnotationClick(annotationRef)}
                        />
                      ))
                    )}
                  </div>
                )}

                {/* Annotation indicator badge */}
                {showAnnotations && currentPageAnnotations.length > 0 && (
                  <div className="absolute top-2 right-2 z-20">
                    <div className="bg-accent text-paper font-mono text-xs px-2 py-1 border border-accent flex items-center gap-1">
                      <span>[§]</span>
                      {currentPageAnnotations.length}
                    </div>
                  </div>
                )}
              </div>
            </Document>
          </div>
        ) : (
          /* =====================================================
             EMPTY STATE - Upload Zone
             ===================================================== */
          <div
            className={`h-full flex flex-col items-center justify-center p-8 transition-all duration-200 ${
              isDragging ? 'bg-accent/5' : ''
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {/* Panel Number */}
            <div className="flex items-center gap-2 mb-6">
              <span className="font-mono text-xs text-accent">[003]</span>
              <span className="font-mono text-xs uppercase">Document Viewer</span>
            </div>

            {/* Upload Zone */}
            <div
              onClick={() => fileInputRef.current?.click()}
              className={`
                no-select group cursor-pointer w-full max-w-xl border-2 border-dashed border-ink p-12
                flex flex-col items-center justify-center gap-4 text-center transition-all duration-200
                ${isDragging
                  ? 'bg-accent/10 border-accent'
                  : 'hover:bg-accent/5'
                }
              `}
            >
              <div className={`
                p-4 border-2 border-ink transition-all duration-200 mb-2
                ${isDragging
                  ? 'bg-accent text-paper'
                  : 'group-hover:bg-accent group-hover:text-paper'
                }
              `}>
                {isProcessing ? (
                  <Loader2 className="h-8 w-8 animate-spin" />
                ) : (
                  <Upload className="h-8 w-8" />
                )}
              </div>

              <div className="space-y-2">
                <h3 className="font-mono text-xl text-ink group-hover:text-accent transition-colors">
                  [{isProcessing ? 'PROCESSING...' : 'UPLOAD DOCUMENT'}]
                </h3>
                <p className="font-serif text-sm text-subtle">
                  Drag & drop your PDF here, or click to browse
                </p>
              </div>

              <div className="flex gap-4 mt-4 font-mono text-xs">
                <span className="px-3 py-1 border border-ink text-subtle">[searchable]</span>
                <span className="px-3 py-1 border border-ink text-subtle">[high-res]</span>
              </div>
            </div>

            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept="application/pdf"
              className="hidden"
            />
          </div>
        )}
      </div>
    </div>
  );
});

EnhancedPDFViewer.displayName = 'EnhancedPDFViewer';

export default EnhancedPDFViewer;

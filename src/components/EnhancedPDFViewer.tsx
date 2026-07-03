// app/components/EnhancedPDFViewer.tsx
import { Loader2, FileUp, Highlighter, Minimize2, RotateCw, ZoomIn, ZoomOut, ChevronLeft, ChevronRight } from "lucide-react";
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

// Highlighter colors — the citation signature (replaces hardcoded orange)
const MARK_FILL = 'var(--mark)';
const MARK_EDGE = 'var(--mark-edge)';

// Small primitive for the duplicated reader-bar button class string.
interface ToolbarButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  active?: boolean;
}
const ToolbarButton = forwardRef<HTMLButtonElement, ToolbarButtonProps>(
  ({ active, className = '', children, ...rest }, ref) => (
    <button
      ref={ref}
      className={`no-select inline-flex items-center justify-center rounded-xs min-w-[44px] min-h-[44px] p-1.5 transition-colors ease-desk disabled:opacity-30 disabled:cursor-not-allowed ${
        active
          ? 'bg-accent text-white'
          : 'text-subtle hover:text-ink hover:bg-desk'
      } ${className}`}
      {...rest}
    >
      {children}
    </button>
  )
);
ToolbarButton.displayName = 'ToolbarButton';

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
            backgroundColor: color || MARK_FILL,
          }}
          onClick={onClick}
          onKeyDown={handleKeyDown}
          className="hover:brightness-105"
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
            border: `2px solid ${color || 'var(--accent)'}`,
            borderRadius: '50%',
            backgroundColor: 'transparent',
          }}
          onClick={onClick}
          onKeyDown={handleKeyDown}
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
            border: `2px solid ${color || 'var(--accent)'}`,
            backgroundColor: color ? `${color}1a` : 'var(--accent-soft)',
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
            height: '2px',
            top: `${bounds.y + bounds.height}%`,
            backgroundColor: color || 'var(--accent)',
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
    onPageChange: () => setPageInputError(null),
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
      return;
    }

    const textLayer = pageContainerRef.current.querySelector('.react-pdf__Page__textContent');
    if (!textLayer) {
      // Retry after a short delay if text layer isn't ready
      setTimeout(() => findTextOnPage(searchText), 300);
      return;
    }

    const textSpans = textLayer.querySelectorAll('span');
    const searchLower = searchText.toLowerCase().trim();
    const searchWords = searchLower.split(/\s+/).filter(w => w.length > 2);
    const rects: {x: number, y: number, width: number, height: number}[] = [];


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

    setHighlightedTextRects(rects);
  }, []);

  // Throttled version to avoid excessive searches during rapid changes
  const throttledFindText = useThrottledCallback(findTextOnPage, 300);

  // Expose methods to parent via ref
  useImperativeHandle(ref, () => ({
    goToPage: (pageNum: number) => {
      goToPage(pageNum);
    },
    setAnnotations: (annotations: AnnotationReference[]) => {
      setLocalAnnotations(annotations);
    },
    clearAnnotations: () => {
      setLocalAnnotations([]);
      setHighlightedTextRects([]);
    },
    highlightText: (pageNum: number, textToFind: string) => {
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
        const response = await fetch(currentPDF, {
          credentials: 'include',
          cache: 'force-cache', // Cache the PDF for better performance
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch PDF: ${response.status} ${response.statusText}`);
        }

        const blob = await response.blob();
        objectUrl = URL.createObjectURL(blob);

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
    <div className="w-full h-full flex flex-col bg-desk relative overflow-hidden">
      {/* =====================================================
          HEADER — clean reader bar
          ===================================================== */}
      {currentPDF && (
        <div className="border-b border-hair bg-paper">
          <div className="flex items-center justify-between px-4 py-2.5">
            {/* Left: document name */}
            <div className="flex items-center gap-2 overflow-hidden min-w-0">
              <h3 className="text-sm font-medium truncate text-ink min-w-0">
                {currentPDF.split('/').pop() || 'Document'}
              </h3>
            </div>

            {/* Right: page counter + actions */}
            <div className="flex items-center gap-1 flex-shrink-0">
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

              <span className="font-mono text-xs text-subtle tabular-nums mr-1">
                {pageNumber} / {numPages || '–'}
              </span>

              {/* Annotation Toggle */}
              {allAnnotations.length > 0 && (
                <ToolbarButton
                  onClick={() => setShowAnnotations(!showAnnotations)}
                  active={showAnnotations}
                  title={showAnnotations ? 'Hide annotations' : 'Show annotations'}
                  aria-label={showAnnotations ? 'Hide annotations' : 'Show annotations'}
                  aria-pressed={showAnnotations}
                >
                  <Highlighter className="h-4 w-4" />
                </ToolbarButton>
              )}

              <ToolbarButton onClick={rotate} title="Rotate page" aria-label="Rotate page">
                <RotateCw className="h-4 w-4" />
              </ToolbarButton>

              {/* Collapse Button - Only show when onCollapse is provided */}
              {onCollapse && (
                <ToolbarButton onClick={onCollapse} title="Hide PDF viewer" aria-label="Hide PDF viewer">
                  <Minimize2 className="h-4 w-4" />
                </ToolbarButton>
              )}
            </div>
          </div>

          {/* Control Bar — zoom + page navigation */}
          <div className="flex items-center justify-center gap-3 px-4 py-1.5 border-t border-hair-soft">
            {/* Zoom Controls */}
            <div className="flex items-center gap-1">
              <ToolbarButton
                onClick={() => setScale(prev => Math.max(0.5, prev - 0.1))}
                title="Zoom out"
                aria-label="Zoom out"
              >
                <ZoomOut className="h-4 w-4" />
              </ToolbarButton>
              <span className="font-mono text-xs w-12 text-center tabular-nums text-subtle">
                {Math.round(scale * 100)}%
              </span>
              <ToolbarButton
                onClick={() => setScale(prev => Math.min(2, prev + 0.1))}
                title="Zoom in"
                aria-label="Zoom in"
              >
                <ZoomIn className="h-4 w-4" />
              </ToolbarButton>
            </div>

            {/* Divider */}
            <div className="w-px h-5 bg-hair"></div>

            {/* Page Navigation */}
            <div className="flex items-center gap-1">
              <ToolbarButton
                onClick={prevPage}
                disabled={!canGoPrev}
                title="Previous page"
                aria-label="Previous page"
              >
                <ChevronLeft className="h-4 w-4" />
              </ToolbarButton>

              <div className="flex items-center gap-1 font-mono text-xs bg-desk rounded-xs px-2 py-1">
                <input
                  ref={pageInputRef}
                  type="text"
                  defaultValue={pageNumber}
                  key={pageNumber}
                  className="w-7 text-center bg-transparent focus:outline-none font-mono text-ink"
                  onChange={handlePageInputChange}
                  onKeyDown={handlePageInputKeyDown}
                  onFocus={(e) => e.target.select()}
                  aria-label="Go to page"
                />
                <span className="text-faint">/ {numPages || '–'}</span>
              </div>

              <ToolbarButton
                onClick={nextPage}
                disabled={!canGoNext}
                title="Next page"
                aria-label="Next page"
              >
                <ChevronRight className="h-4 w-4" />
              </ToolbarButton>
            </div>

            {/* Error Toast */}
            {pageInputError && (
              <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 px-3 py-1 bg-accent-soft text-accent-ink rounded-xs text-xs">
                {pageInputError}
              </div>
            )}
          </div>
        </div>
      )}

      {/* =====================================================
          CONTENT AREA — the page on the desk
          ===================================================== */}
      <div
        className="flex-1 overflow-auto relative z-10 scrollbar-thin bg-desk"
        ref={containerRef}
      >
        {loadingPdf ? (
          <div className="flex flex-col items-center justify-center h-full gap-6 p-8">
            {/* PDF skeleton — page-shaped */}
            <div className="w-full max-w-md space-y-3">
              <div className="h-80 bg-surface rounded-sm shadow-page relative overflow-hidden p-8 space-y-3">
                <div className="absolute top-8 left-8 right-8 space-y-2">
                  <div className="h-3 bg-hair animate-pulse rounded-xs w-3/4" />
                  <div className="h-3 bg-hair animate-pulse rounded-xs delay-75 w-full" />
                  <div className="h-3 bg-hair animate-pulse rounded-xs delay-100 w-5/6" />
                </div>
                <div className="absolute top-20 left-8 right-8 space-y-2">
                  <div className="h-3 bg-hair animate-pulse rounded-xs w-full" />
                  <div className="h-3 bg-hair animate-pulse rounded-xs delay-75 w-2/3" />
                </div>
                <div className="absolute top-32 left-8 right-8 space-y-2">
                  <div className="h-3 bg-hair animate-pulse rounded-xs w-4/5" />
                  <div className="h-3 bg-hair animate-pulse rounded-xs delay-75 w-full" />
                  <div className="h-3 bg-hair animate-pulse rounded-xs delay-100 w-3/4" />
                </div>
              </div>
              <div className="flex justify-center">
                <div className="h-5 w-24 bg-hair animate-pulse rounded-xs" />
              </div>
            </div>
            <span className="text-sm text-faint">Loading…</span>
          </div>
        ) : pdfError ? (
          <div className="flex flex-col items-center justify-center h-64 gap-3">
            <span className="text-sm font-medium text-ink">Failed to load PDF</span>
            <span className="font-mono text-xs text-faint">{pdfError}</span>
            <button onClick={() => fileInputRef.current?.click()} className="btn btn-primary mt-2">
              Try again
            </button>
          </div>
        ) : pdfUrl ? (
          <div className="flex justify-center min-h-full p-6">
            <Document
              file={pdfUrl}
              onLoadSuccess={onDocumentLoadSuccess}
              loading={
                <div className="flex flex-col items-center justify-center h-64 gap-4">
                  <div className="w-64 space-y-2">
                    <div className="h-48 bg-surface rounded-sm shadow-card p-4 space-y-2">
                      <div className="h-2 bg-hair animate-pulse rounded-xs w-full" />
                      <div className="h-2 bg-hair animate-pulse rounded-xs delay-75 w-4/5" />
                      <div className="h-2 bg-hair animate-pulse rounded-xs delay-100 w-11/12" />
                    </div>
                    <div className="h-3 bg-hair animate-pulse rounded-xs w-20 mx-auto" />
                  </div>
                  <span className="text-sm text-faint">Loading…</span>
                </div>
              }
              error={
                <div className="flex flex-col items-center justify-center h-64 gap-2">
                  <span className="text-sm font-medium text-ink">Failed to load PDF</span>
                  <button onClick={() => fileInputRef.current?.click()} className="btn btn-primary mt-2">
                    Try again
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
                className="relative bg-surface shadow-page rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-desk paper-grain"
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
                    {/* Highlight rectangles from text search — the calm mark-fade sweep */}
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
                          className="absolute pointer-events-auto cursor-pointer mark-fade"
                          style={{
                            left: rect.x,
                            top: rect.y,
                            width: rect.width,
                            height: rect.height,
                            backgroundColor: MARK_FILL,
                            border: `1px solid ${MARK_EDGE}`,
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

                    {/* Annotation shapes from annotation data */}
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
                    <div className="bg-accent text-white rounded-full font-mono text-[10px] px-2 py-0.5 flex items-center gap-1 shadow-card">
                      <Highlighter className="h-2.5 w-2.5" />
                      {currentPageAnnotations.length}
                    </div>
                  </div>
                )}
              </div>
            </Document>
          </div>
        ) : (
          /* =====================================================
             EMPTY STATE — upload zone
             ===================================================== */
          <div
            className={`h-full flex flex-col items-center justify-center p-8 transition-colors ease-desk ${
              isDragging ? 'bg-accent-soft' : ''
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div
              onClick={() => fileInputRef.current?.click()}
              className={`
                no-select group cursor-pointer w-full max-w-xl rounded-lg p-12
                flex flex-col items-center justify-center gap-4 text-center transition-colors ease-desk
                border-2 border-dashed
                ${isDragging
                  ? 'bg-surface border-accent'
                  : 'border-hair hover:border-accent hover:bg-surface'
                }
              `}
            >
              <div className={`
                p-4 rounded-full transition-colors ease-desk mb-1
                ${isDragging
                  ? 'bg-accent text-white'
                  : 'bg-desk text-accent group-hover:bg-accent group-hover:text-white'
                }
              `}>
                {isProcessing ? (
                  <Loader2 className="h-7 w-7 animate-spin" />
                ) : (
                  <FileUp className="h-7 w-7" />
                )}
              </div>

              <div className="space-y-1.5">
                <h3 className="text-xl font-medium text-ink">
                  {isProcessing ? 'Processing…' : 'Upload a PDF'}
                </h3>
                <p className="text-sm text-subtle">
                  Drag &amp; drop your PDF here, or click to browse
                </p>
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

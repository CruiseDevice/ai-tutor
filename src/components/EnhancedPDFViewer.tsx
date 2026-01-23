// app/components/EnhancedPDFViewer.tsx
import { ChevronLeft, ChevronRight, Loader2, Upload, ZoomIn, ZoomOut, RotateCw, Search, Eye, EyeOff } from "lucide-react";
import { useRef, useState, useCallback, useImperativeHandle, forwardRef, useEffect, useMemo } from "react";
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/TextLayer.css';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import type { PDFAnnotation, AnnotationReference } from '@/types/annotations';

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

  switch (type) {
    case 'highlight':
      return (
        <div
          style={{
            ...baseStyle,
            backgroundColor: color || 'rgba(255, 235, 59, 0.4)',
            borderRadius: '2px',
          }}
          onClick={onClick}
          className="hover:brightness-110 hover:shadow-lg"
        />
      );

    case 'circle':
      return (
        <div
          style={{
            ...baseStyle,
            border: `3px solid ${color || 'rgba(33, 150, 243, 0.8)'}`,
            borderRadius: '50%',
            backgroundColor: 'transparent',
          }}
          onClick={onClick}
          className="hover:border-4 hover:shadow-lg animate-pulse"
        />
      );

    case 'box':
      return (
        <div
          style={{
            ...baseStyle,
            border: `3px solid ${color || 'rgba(76, 175, 80, 0.8)'}`,
            backgroundColor: color?.replace('0.8', '0.1') || 'rgba(76, 175, 80, 0.1)',
            borderRadius: '4px',
          }}
          onClick={onClick}
          className="hover:border-4 hover:shadow-lg"
        />
      );

    case 'underline':
      return (
        <div
          style={{
            ...baseStyle,
            height: '3px',
            top: `${bounds.y + bounds.height}%`,
            backgroundColor: color || 'rgba(244, 67, 54, 0.8)',
          }}
          onClick={onClick}
          className="hover:h-1"
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
  currentPDF: string | null;
  onFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  fileInputRef: React.RefObject<HTMLInputElement>;
  annotations?: AnnotationReference[];
  onAnnotationClick?: (annotation: AnnotationReference) => void;
}

const EnhancedPDFViewer = forwardRef<PDFViewerRef, EnhancedPDFViewerProps>(({
  currentPDF,
  onFileUpload,
  fileInputRef: externalFileInputRef,
  annotations: externalAnnotations = [],
  onAnnotationClick,
}, ref) => {

  const internalFileInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = externalFileInputRef || internalFileInputRef;
  const pageInputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const pageContainerRef = useRef<HTMLDivElement>(null);

  const [pageNumber, setPageNumber] = useState(1);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [scale, setScale] = useState(1.0);
  const [rotation, setRotation] = useState(0);
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

  // Touch swipe gesture state for page navigation
  const [touchStart, setTouchStart] = useState<number | null>(null);
  const [touchEnd, setTouchEnd] = useState<number | null>(null);

  // Combine external and local annotations
  const allAnnotations = useMemo(() => [...externalAnnotations, ...localAnnotations], [externalAnnotations, localAnnotations]);

  // Get annotations for current page
  const currentPageAnnotations = useMemo(
    () => allAnnotations.filter(a => a.pageNumber === pageNumber),
    [allAnnotations, pageNumber]
  );

  const goToPage = useCallback((pageNum: number) => {
    if(pageNum >= 1 && pageNum <= (numPages || 1)) {
      setPageNumber(pageNum);
      setPageInputError(null);
    }
  }, [numPages]);

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
      // Text highlighting will be handled by findTextOnPage after page renders
      setTimeout(() => findTextOnPage(textToFind), 800);
    }
  }), [goToPage, findTextOnPage]);

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
      textTargets.forEach(text => {
        setTimeout(() => findTextOnPage(text), 300);
      });
    } else {
      setHighlightedTextRects([]);
    }
  }, [currentPageAnnotations, showAnnotations, findTextOnPage, pageNumber]);

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
      textTargets.forEach(text => {
        setTimeout(() => findTextOnPage(text), 500);
      });
    }
  }, [currentPageAnnotations, showAnnotations, findTextOnPage]);

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
    // Clear input after successful jump or keep it?
    // Usually better to keep it sync'd or clear it.
    if (pageInputRef.current) {
      pageInputRef.current.value = pageNum.toString();
      pageInputRef.current.blur();
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
    setPageNumber(1);
  }

  const rotate = () => {
    setRotation((prev) => (prev + 90) % 360);
  };

  // Touch swipe gesture handlers for page navigation
  const minSwipeDistance = 50; // Minimum horizontal distance to trigger page change

  const handleTouchStart = (e: React.TouchEvent) => {
    setTouchStart(e.touches[0].clientX);
    setTouchEnd(null);
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    setTouchEnd(e.touches[0].clientX);
  };

  const handleTouchEnd = () => {
    if (touchStart === null || touchEnd === null) return;

    const distance = touchStart - touchEnd;
    const isLeftSwipe = distance > minSwipeDistance;
    const isRightSwipe = distance < -minSwipeDistance;

    if (isLeftSwipe && pageNumber < (numPages || 1)) {
      // Swiped left → next page
      goToPage(pageNumber + 1);
    } else if (isRightSwipe && pageNumber > 1) {
      // Swiped right → previous page
      goToPage(pageNumber - 1);
    }

    // Reset for next gesture
    setTouchStart(null);
    setTouchEnd(null);
  };

  return (
    <div className="w-full h-full flex flex-col bg-slate-50/50 relative overflow-hidden group">
      {/* Background Pattern */}
      <div className="absolute inset-0 z-0 opacity-[0.03]"
           style={{
             backgroundImage: 'radial-gradient(#64748b 1px, transparent 1px)',
             backgroundSize: '24px 24px'
           }}
      />

      {/* Control Bar - Floating Glass Effect */}
      {currentPDF && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 w-[90%] max-w-4xl">
          <div className="bg-white/80 backdrop-blur-md border border-white/20 shadow-lg rounded-2xl px-4 py-2 flex items-center justify-between gap-4 transition-all duration-300 hover:bg-white/95 hover:shadow-xl ring-1 ring-black/5">

            {/* Left: Zoom Controls */}
            <div className="flex items-center gap-1 bg-slate-100/50 rounded-lg p-1">
              <button
                onClick={() => setScale(prev => Math.max(0.5, prev - 0.1))}
                className="no-select no-tap-highlight p-1.5 rounded-md hover:bg-white hover:shadow-sm text-slate-600 transition-all min-w-[44px] min-h-[44px] flex items-center justify-center"
                title="Zoom Out"
              >
                <ZoomOut size={16} />
              </button>
              <span className="text-xs font-medium text-slate-600 w-12 text-center select-none">
                {Math.round(scale * 100)}%
              </span>
              <button
                onClick={() => setScale(prev => Math.min(2, prev + 0.1))}
                className="no-select no-tap-highlight p-1.5 rounded-md hover:bg-white hover:shadow-sm text-slate-600 transition-all min-w-[44px] min-h-[44px] flex items-center justify-center"
                title="Zoom In"
              >
                <ZoomIn size={16} />
              </button>
            </div>

            {/* Center: Page Navigation */}
            <div className="flex items-center gap-3">
              <button
                onClick={() => goToPage(pageNumber - 1)}
                disabled={pageNumber <= 1}
                className="no-select no-tap-highlight p-2 rounded-full hover:bg-slate-100 disabled:opacity-30 disabled:hover:bg-transparent text-slate-700 transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
              >
                <ChevronLeft size={20}/>
              </button>

              <div className="flex items-center gap-2 text-sm font-medium text-slate-600 bg-slate-100/50 px-3 py-1.5 rounded-lg border border-transparent focus-within:border-blue-200 focus-within:bg-white transition-all">
                <span className="text-slate-400">Page</span>
                <input
                  ref={pageInputRef}
                  type="text"
                  defaultValue={pageNumber}
                  key={pageNumber} // Force re-render on page change
                  className="w-8 text-center bg-transparent focus:outline-none text-slate-900 font-semibold"
                  onChange={handlePageInputChange}
                  onKeyDown={handlePageInputKeyDown}
                  onFocus={(e) => e.target.select()}
                  aria-label="Go to page"
                />
                <span className="text-slate-400">/ {numPages || '-'}</span>
              </div>

              <button
                onClick={() => goToPage(pageNumber + 1)}
                disabled={pageNumber >= (numPages || 1)}
                className="no-select no-tap-highlight p-2 rounded-full hover:bg-slate-100 disabled:opacity-30 disabled:hover:bg-transparent text-slate-700 transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
              >
                <ChevronRight size={20}/>
              </button>
            </div>

            {/* Right: Tools */}
            <div className="flex items-center gap-2">
              {/* Annotation Toggle */}
              {allAnnotations.length > 0 && (
                <button
                  onClick={() => setShowAnnotations(!showAnnotations)}
                  className={`no-select no-tap-highlight p-2 rounded-lg transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center ${
                    showAnnotations
                      ? 'bg-yellow-100 text-yellow-700 hover:bg-yellow-200'
                      : 'hover:bg-slate-100 text-slate-400'
                  }`}
                  title={showAnnotations ? 'Hide Annotations' : 'Show Annotations'}
                >
                  {showAnnotations ? <Eye size={18} /> : <EyeOff size={18} />}
                </button>
              )}
              <button
                onClick={rotate}
                className="no-select no-tap-highlight p-2 rounded-lg hover:bg-slate-100 text-slate-600 transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
                title="Rotate Page"
              >
                <RotateCw size={18} />
              </button>
            </div>

            {/* Error Toast */}
            {pageInputError && (
              <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 px-3 py-1 bg-red-100 text-red-600 text-xs rounded-full shadow-lg border border-red-200 animate-in fade-in slide-in-from-top-1">
                {pageInputError}
              </div>
            )}
          </div>
        </div>
      )}

      {/* PDF Content Area */}
      <div
        className="flex-1 overflow-auto relative z-10 pt-20 pb-8 px-4 scrollbar-thin scrollbar-thumb-slate-200 scrollbar-track-transparent"
        ref={containerRef}
      >
        {loadingPdf ? (
          <div className="flex flex-col items-center justify-center h-64 gap-3">
            <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />
            <span className="text-sm text-slate-500 font-medium">Loading Document...</span>
          </div>
        ) : pdfError ? (
          <div className="flex flex-col items-center justify-center h-64 gap-2 text-red-500">
            <div className="p-3 bg-red-50 rounded-full">
              <Upload size={24} />
            </div>
            <span className="font-medium">Failed to load PDF</span>
            <span className="text-sm text-red-400">{pdfError}</span>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="no-tap-highlight text-xs text-blue-500 hover:underline mt-2 min-h-[44px]"
            >
              Try uploading again
            </button>
          </div>
        ) : pdfUrl ? (
          <div className="flex justify-center min-h-full">
            <Document
              file={pdfUrl}
              onLoadSuccess={onDocumentLoadSuccess}
              loading={
                <div className="flex flex-col items-center justify-center h-64 gap-3">
                  <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />
                  <span className="text-sm text-slate-500 font-medium">Loading Document...</span>
                </div>
              }
              error={
                <div className="flex flex-col items-center justify-center h-64 gap-2 text-red-500">
                  <div className="p-3 bg-red-50 rounded-full">
                    <Upload size={24} />
                  </div>
                  <span className="font-medium">Failed to load PDF</span>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="no-tap-highlight text-xs text-blue-500 hover:underline mt-2 min-h-[44px]"
                  >
                    Try uploading again
                  </button>
                </div>
              }
              className="pdf-document transition-opacity duration-300 ease-in-out"
            >
              <div
                ref={pageContainerRef}
                onTouchStart={handleTouchStart}
                onTouchMove={handleTouchMove}
                onTouchEnd={handleTouchEnd}
                className="relative transition-all duration-300 ease-out shadow-2xl shadow-slate-200/50 rounded-sm overflow-hidden"
                style={{
                  transform: `scale(${1})`, // Scale handled by react-pdf prop usually, but we can wrap for effects
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
                    {highlightedTextRects.map((rect, idx) => (
                      <div
                        key={`highlight-rect-${idx}`}
                        className="absolute pointer-events-auto cursor-pointer transition-all duration-200 animate-pulse"
                        style={{
                          left: rect.x,
                          top: rect.y,
                          width: rect.width,
                          height: rect.height,
                          backgroundColor: 'rgba(255, 235, 59, 0.5)',
                          border: '2px solid rgba(255, 193, 7, 0.8)',
                          borderRadius: '2px',
                          boxShadow: '0 0 8px rgba(255, 235, 59, 0.6)',
                        }}
                        onClick={() => {
                          if (onAnnotationClick && currentPageAnnotations[0]) {
                            onAnnotationClick(currentPageAnnotations[0]);
                          }
                        }}
                      />
                    ))}

                    {/* Render annotation shapes from annotation data */}
                    {currentPageAnnotations.map((annotationRef) =>
                      annotationRef.annotations.map((annotation) => (
                        <AnnotationShape
                          key={annotation.id}
                          annotation={annotation}
                          onClick={() => onAnnotationClick?.(annotationRef)}
                        />
                      ))
                    )}
                  </div>
                )}

                {/* Annotation indicator badge */}
                {showAnnotations && currentPageAnnotations.length > 0 && (
                  <div className="absolute top-2 right-2 z-20">
                    <div className="bg-yellow-400 text-yellow-900 text-xs font-bold px-2 py-1 rounded-full shadow-md flex items-center gap-1 animate-bounce">
                      <Eye size={12} />
                      {currentPageAnnotations.length} annotation{currentPageAnnotations.length > 1 ? 's' : ''}
                    </div>
                  </div>
                )}
              </div>
            </Document>
          </div>
        ) : (
          <div
            className={`h-full flex flex-col items-center justify-center p-8 transition-all duration-300 ${
              isDragging ? 'scale-105 bg-blue-50/50' : ''
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div
              onClick={() => fileInputRef.current?.click()}
              className={`
                no-select no-tap-highlight group cursor-pointer w-full max-w-xl border-2 border-dashed rounded-3xl p-12
                flex flex-col items-center justify-center gap-4 text-center transition-all duration-300
                ${isDragging
                  ? 'border-blue-500 bg-blue-50/50 shadow-lg shadow-blue-100'
                  : 'border-slate-200 hover:border-blue-400 hover:bg-white hover:shadow-xl hover:shadow-slate-100'
                }
              `}
            >
              <div className={`
                p-4 rounded-full transition-all duration-300 mb-2
                ${isDragging
                  ? 'bg-blue-100 text-blue-600 scale-110'
                  : 'bg-slate-50 text-slate-400 group-hover:bg-blue-50 group-hover:text-blue-500 group-hover:scale-110'
                }
              `}>
                {isProcessing ? (
                  <Loader2 className="h-10 w-10 animate-spin" />
                ) : (
                  <Upload className="h-10 w-10" />
                )}
              </div>

              <div className="space-y-1">
                <h3 className="text-xl font-semibold text-slate-700 group-hover:text-blue-600 transition-colors">
                  {isProcessing ? 'Processing...' : 'Upload Course Material'}
                </h3>
                <p className="text-sm text-slate-500">
                  Drag and drop your PDF here, or click to browse
                </p>
              </div>

              <div className="flex gap-4 mt-4 opacity-50 text-xs text-slate-400">
                <span className="flex items-center gap-1"><Search size={12}/> Searchable</span>
                <span className="flex items-center gap-1"><ZoomIn size={12}/> High Res</span>
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

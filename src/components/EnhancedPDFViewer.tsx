// app/components/EnhancedPDFViewer.tsx
import { ChevronLeft, ChevronRight, Loader2, Upload, ZoomIn, ZoomOut, RotateCw, Search } from "lucide-react";
import { useRef, useState } from "react";
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/TextLayer.css';
import 'react-pdf/dist/Page/AnnotationLayer.css';

// Initialize pdfjs worker
pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.mjs`;

interface EnhancedPDFViewerProps {
  currentPDF: string | null;
  onFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  fileInputRef: React.RefObject<HTMLInputElement>;
}

export default function EnhancedPDFViewer({
  currentPDF,
  onFileUpload,
  fileInputRef: externalFileInputRef,
}: EnhancedPDFViewerProps){

  const internalFileInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = externalFileInputRef || internalFileInputRef;
  const pageInputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [pageNumber, setPageNumber] = useState(1);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [scale, setScale] = useState(1.0);
  const [rotation, setRotation] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [pageInputError, setPageInputError] = useState<string | null>(null);

  const goToPage = (pageNum: number) => {
    if(pageNum >= 1 && pageNum <= (numPages || 1)) {
      setPageNumber(pageNum);
      setPageInputError(null);
    }
  }

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
                className="p-1.5 rounded-md hover:bg-white hover:shadow-sm text-slate-600 transition-all"
                title="Zoom Out"
              >
                <ZoomOut size={16} />
              </button>
              <span className="text-xs font-medium text-slate-600 w-12 text-center select-none">
                {Math.round(scale * 100)}%
              </span>
              <button
                onClick={() => setScale(prev => Math.min(2, prev + 0.1))}
                className="p-1.5 rounded-md hover:bg-white hover:shadow-sm text-slate-600 transition-all"
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
                className="p-2 rounded-full hover:bg-slate-100 disabled:opacity-30 disabled:hover:bg-transparent text-slate-700 transition-colors"
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
                className="p-2 rounded-full hover:bg-slate-100 disabled:opacity-30 disabled:hover:bg-transparent text-slate-700 transition-colors"
              >
                <ChevronRight size={20}/>
              </button>
            </div>

            {/* Right: Tools */}
            <div className="flex items-center gap-2">
              <button
                onClick={rotate}
                className="p-2 rounded-lg hover:bg-slate-100 text-slate-600 transition-colors"
                title="Rotate Page"
              >
                <RotateCw size={18} />
              </button>
              <div className="h-4 w-[1px] bg-slate-200 mx-1" />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-blue-600 bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors"
              >
                <Upload size={14} />
                Replace
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
        {currentPDF ? (
          <div className="flex justify-center min-h-full">
            <Document
              file={currentPDF}
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
                    className="text-xs text-blue-500 hover:underline mt-2"
                  >
                    Try uploading again
                  </button>
                </div>
              }
              className="pdf-document transition-opacity duration-300 ease-in-out"
            >
              <div
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
                />
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
                group cursor-pointer w-full max-w-xl border-2 border-dashed rounded-3xl p-12
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
}
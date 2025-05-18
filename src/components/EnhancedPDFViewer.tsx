// app/components/EnhancedPDFViewer.tsx
import { ChevronLeft, ChevronRight, Loader, Loader2, Upload } from "lucide-react";
import { useRef, useState } from "react";
import {Document, Page, pdfjs} from 'react-pdf';
import 'react-pdf/dist/Page/TextLayer.css';

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

  const [pageNumber, setPageNumber] = useState(1);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [scale, setScale] = useState(1.0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [pageInputError, setPageInputError] = useState<string | null>(null);

  const goToPage = (pageNum: number) => {
    if(pageNum >= 1 && pageNum <= (numPages || 1)) {
      setPageNumber(pageNum)
    }
  }

  const handlePageInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setPageInputError(null);
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
      setPageInputError('Please enter a valid page number');
      return;
    }

    if (pageNum < 1 || pageNum > (numPages || 1)) {
      setPageInputError(`Please enter a page between 1 and ${numPages || 1}`);
      return;
    }

    goToPage(pageNum);
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    setIsProcessing(true);
    try {
      await onFileUpload(e);
    } finally {
      setIsProcessing(false);
    }
  }
  const onDocumentLoadSuccess = ({numPages}: {numPages: number}) => {
    setNumPages(numPages);
  }

  return (
    <div className="w-full h-full bg-white border-r border-gray-200 flex flex-col min-h-0">
      {/* control headers */}
      {currentPDF && (
        <div className="flex justify-between items-center p-2 border-b border-gray-200">
          <div className="flex items-center space-x-2">
            <button
              onClick={() => goToPage(pageNumber - 1)}
              disabled={pageNumber <= 1}
              className="p-1 rounded border border-gray-300 disabled:opacity-30"
            >
              <ChevronLeft size={16}/>
            </button>
            <span className="text-sm">
              Page {pageNumber} of {numPages || '?'}
            </span>
            <button
              onClick={() => goToPage(pageNumber + 1)}
              disabled={pageNumber >= (numPages || 1)}
              className="p-1 rounded border border-gray-300 disabled:opacity-30"
            >
              <ChevronRight size={16}/>
            </button>

            {/* page jump input */}
            <div className="relative ml-4">
              <input 
                ref={pageInputRef}
                type="text"
                placeholder="Go to page..."
                className="w-32 px-2 py-1 text-sm border border-gray-300 rounded"
                onChange={handlePageInputChange}
                onKeyDown={handlePageInputKeyDown}
                aria-label="Go to page"
              />
              <button
                onClick={handlePageJump}
                className="ml-1 px-2 py-2 text-sm bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Go
              </button>
              {pageInputError && (
                <div className="absolute left-0 mt-1 text-xs text-red-500 w-full">
                  {pageInputError}
                </div>
              )}
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button 
              onClick={() => setScale(prev => Math.max(0.5, prev - 0.1))}
              className="p-1 rounded border border-gray-300">
              -
            </button>
            <span className="text-sm">{Math.round(scale * 100)}%</span>
            <button 
              onClick={() => setScale(prev => Math.min(2, prev + 0.1))}
              className="p-1 rounded border border-gray-300">
              +
            </button>
          </div>
        </div>
      )}
      {/* PDF viewer content */}
      <div className="flex-1 overflow-auto">
        {currentPDF ? (
          <div className="flex justify-center">
            <Document 
              file={currentPDF}
              onLoadSuccess={onDocumentLoadSuccess}
              loading={<div><Loader2 className="animate-spin"/></div>}
              error={<div className="p-10 text-red-500">Failed to load PDF</div>}
              className="pdf-document"
            >
            <Page 
              pageNumber={pageNumber}
              scale={scale}
              renderAnnotationLayer={false}
              renderTextLayer={true}
              className="pdf-page"
            />
          </Document>
          </div>
        ) : (
          <div className="h-full flex items-center justify-center">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-2 bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600"
            >
              {isProcessing ? (
                <>
                  <Loader className="h-5 w-5 animate-spin"/>
                  Processing PDF...
                </>
              ) : (
                <>
                  <Upload size={20}/>
                  Upload PDF
                </>
              )}
            </button>
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
      {/* Styling for highlights */}
      <style jsx global>{`
          .pdf-document {
            display: inline-block;
          }
          .pdf-page {
            margin-bottom: 10px;
            position: relative;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
          }
        `}
      </style>
    </div>
  );
}
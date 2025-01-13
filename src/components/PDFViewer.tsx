import { Loader2, Upload } from "lucide-react";
import React, { useRef, useState } from "react"

interface PDFViewerProps {
  currentPDF: string | null;
  onFileUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

export default function PDFViewer({
  currentPDF, 
  onFileUpload}: PDFViewerProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    setIsProcessing(true);
    try {
      await onFileUpload(e);
    } finally {
      setIsProcessing(false);
    }
  };
  return (
    <div className="w-1/2 bg-white border-r border-gray-200 p-4">
      {currentPDF ? (
        <embed 
          src={currentPDF}
          type="application/pdf"
          className="w-full h-full"
        />
      ) : (
        <div className="h-full flex items-center justify-center">
          <button 
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-2 bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600"
          >
            {isProcessing ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin"/>
                Processing PDF...
              </>
            ): (
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
  )
}
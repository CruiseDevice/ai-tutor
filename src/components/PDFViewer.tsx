import { Upload } from "lucide-react";
import React, { useRef } from "react"

interface PDFViewerProps {
  currentPDF: string | null;
  onFileUpload: (file: File) => void;
}

export default function PDFViewer({currentPDF, onFileUpload}: PDFViewerProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if(file && file.type === 'application/pdf') {
      onFileUpload(file);
    }
  }
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
            onClick={() => fileInputRef.current.click()}
            className="flex items-center gap-2 bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600"
          >
            <Upload size={20}/>
            Upload PDF
          </button>
          <input 
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="application/pdf"
            className="hidden"
          />
        </div>
      )}
    </div>
  )
}
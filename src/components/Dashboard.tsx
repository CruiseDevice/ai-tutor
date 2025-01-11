// app/components/Dashboard.tsx
"use client";

import { useEffect, useState } from "react"
import PDFViewer from "./PDFViewer";
import ChatInterface from "./ChatInterface";

// message interface for type safety
interface Message {
  role: 'user' | 'assistant';
  content: string
}

export default function Dashboard () {
  const [currentPDF, setCurrentPDF] = useState('');
  const [documentId, setDocumentId] = useState('');
  const [userId, setUserId] = useState('');
  const [error, setError] = useState<string | null>(null);

  // fetch user ID on component mount
  useEffect(() => {
    const fetchUserId = async () => {
      try {
        const response = await fetch('/api/auth/user');
        const data = await response.json();
        if(response.ok) {
          console.log('Fetched user ID: ', data.id);
          setUserId(data.id);
        } else {
          console.error('Failed to fetch user: ', data);
        }
        console.log(data)
      } catch (error) {
        console.error('Error fetching user:', error);
      }
    };
    fetchUserId();
  }, []);

  // log state changes
  useEffect(() => {
    console.log('Current state: ', {userId, documentId, currentPDF});
  }, [userId, documentId, currentPDF]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
   // handlefileupload function triggered
    const file = e.target.files?.[0];
    console.log(file);

    if(!file) return;

    if(file.type !== 'application/pdf') {
      setError('Please upload a PDF file');
      return;
    }

    //  validate file size(e.g., 10MB limit)
    const MAX_SIZE = 10 * 1024 * 1024;
    if (file.size > MAX_SIZE) {
      setError('File size should be less than 10MB');
      return;
    }

    // create a temporary URL for the uploaded file
    const fileURL = URL.createObjectURL(file);
    setCurrentPDF(fileURL);

    // create form data
    const formData = new FormData();
    formData.append('file', file);

    try{
      // upload file
      const response = await fetch('/api/documents', {
        method: 'POST',
        body: formData,
      });
  
      if(!response.ok) {
        throw new Error('Failed to upload document');
      }
  
      const data = await response.json();
      setCurrentPDF(data.url);
      setDocumentId(data.id);
    } catch (error) {
      console.error('Upload error:', error);
      setError('Failed to upload document');
    }
  }

  const handleVoiceRecord = () => {
    // implement voice recording logic later
    console.log('Voice recording toggled');
  }
  if (error) {
    return <div className="p-4 text-red-500">{error}</div>
  }
  return (
    <div className="h-screen flex">
      {/* PDF Viewer Section */}
      <PDFViewer 
        currentPDF={currentPDF}
        onFileUpload={handleFileUpload}
      />
      {/* Chat Section */}
      <ChatInterface
        documentId={documentId}
        userId={userId}
        onVoiceRecord={handleVoiceRecord}
      />
    </div>
  )
}

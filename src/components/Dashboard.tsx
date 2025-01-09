// app/components/Dashboard.tsx
"use client";

import { Mic, Send, Upload } from "lucide-react";
import { useRef, useState } from "react"
import PDFViewer from "./PDFViewer";
import ChatInterface from "./ChatInterface";

// message interface for type safety
interface Message {
  role: 'user' | 'assistant';
  content: string
}

export default function Dashboard () {
  const [currentPDF, setCurrentPDF] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [error, setError] = useState<string | null>(null);

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

    // create form data
    const formData = new FormData();
    formData.append('file', file);

    // upload file
    const response = await fetch('/api/documents', {
      method: 'POST',
      body: formData,
    });
    console.log(response);
  }

  const handleSendMessage = (e) => {
    // send message logic
    console.log('Sendm message function triggered')
  }

  const handleVoiceRecord = () => {
    // implement voice recording logic later
    console.log('Voice recording toggled');
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
        messages={messages}
        onSendMessage={handleSendMessage}
        onVoiceRecord={handleVoiceRecord}
      />
    </div>
  )
}

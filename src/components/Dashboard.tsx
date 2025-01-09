// app/components/Dashboard.tsx
"use client";

import { Mic, Send, Upload } from "lucide-react";
import { useRef, useState } from "react"
import PDFViewer from "./PDFViewer";
import ChatInterface from "./ChatInterface";

export default function Dashboard () {
  const [currentPDF, setCurrentPDF] = useState(null);
  const [messages, setMessages] = useState<Message[]>([]);

  const handleFileUpload = (e) => {
   // handlefileupload function triggered
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

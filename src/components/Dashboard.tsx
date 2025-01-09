// app/components/Dashboard.tsx
"use client";

import { Mic, Send, Upload } from "lucide-react";
import { useRef, useState } from "react"

export default function Dashboard () {
  const [currentPDF, setCurrentPDF] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const[isRecording, setIsRecording] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if(file && file.type === 'application/pdf') {
      setCurrentPDF(URL.createObjectURL(file))
    }
  }

  const handleSendMessage = (e) => {
    e.preventDefault();
  }

  const handleVoiceRecord = () => {
    setIsRecording(!isRecording);
  }
  return (
    <div className="h-screen flex">
      {/* PDF Viewer Section */}
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
              onChange={handleFileUpload}
              accept="application/pdf"
              className="hidden"
            />
          </div>
        )}
      </div>
      {/* Chat Section */}
      <div className="w-1/2 flex flex-col bg-gray-50">
        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${
                message.role==='user' ? 'justify-end' : 'justify-start'
              }`}
              >
                <div 
                  className={`max-w-[80%] p-3 rounded-lg ${
                    message.role === 'user'
                    ? 'bg-blue-500 text-white'
                    : 'bg-white border border-gray-200'
                  }`}
                  >
                    {message.content}
                </div>
            </div>
          ))}
        </div>
        {/* chat input */}
        <div className="border-t border-gray-200 p-4">
          <form
            onSubmit={handleSendMessage}
            className="flex gap-2">
            <button 
              type="button"
              onClick={handleVoiceRecord}
              className={`p-2 rounded-full ${
                isRecording ? 'bg-red-500' : 'bg-gray-200'
              } hover:opacity-80`}>
                <Mic size={20} className={isRecording ? 'text-white' : 'text-gray-600'}/>
              </button>
              <input 
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Ask about the document..."
                className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                type="submit"
                disabled={!inputMessage.trim()}
                className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Send size={20}/>
              </button>
          </form>
        </div>
      </div>
    </div>
  )
}

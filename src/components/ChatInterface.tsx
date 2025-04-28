// app/components/ChatInterface.tsx
import { Mic, Send, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => Promise<void>;
  onVoiceRecord: () => void;
  isConversationSelected: boolean;
}

export default function ChatInterface({
  messages,
  onSendMessage,
  onVoiceRecord,
  isConversationSelected
}: ChatInterfaceProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const errorTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);

  const messageEndRef = useRef<HTMLDivElement>(null);

  // Clear error after 5 seconds
  useEffect(() => {
    if (error) {
      if (errorTimeoutRef.current) {
        clearTimeout(errorTimeoutRef.current);
      }
      errorTimeoutRef.current = setTimeout(() => {
        setError(null);
      }, 5000);
    }
    return () => {
      if (errorTimeoutRef.current) {
        clearTimeout(errorTimeoutRef.current);
      }
    };
  }, [error]);

  // scroll to bottom when messages change
  useEffect(() => {
    messageEndRef.current?.scrollIntoView({behavior: 'smooth'});
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading || !isConversationSelected) return;

    setIsLoading(true);
    setError(null);

    try {
      setInputMessage('');
      await onSendMessage(inputMessage);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to send message. Please try again.';
      setError(errorMessage);
      console.error('Error sending message: ', error);
    } finally {
      setIsLoading(false);
    }
  }

  const handleVoiceRecord = () => {
    try {
      setIsRecording(!isRecording);
      onVoiceRecord();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to start voice recording. Please try again.';
      setError(errorMessage);
      console.error('Error with voice recording: ', error);
    }
  }

  return (
    <div className="w-1/2 h-full flex flex-col bg-gray-50 ">
      {/* Error Message */}
      {error && (
        <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 m-4 rounded relative">
          <button
            onClick={() => setError(null)}
            className="absolute right-2 top-2 text-red-700 hover:text-red-900"
          >
            <X size={16} />
          </button>
          <p>{error}</p>
        </div>
      )}
      
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && !isConversationSelected && (
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            <p>Upload a PDF or select a conversation to start chatting</p>
          </div>
        )}
        {messages.length === 0 && isConversationSelected && (
          <div className="flex h-full items-center justify-center text-gray-500">
            <p>No messages yet. Start the conversation!</p>
          </div>
        )}
        {messages.map((message) => (
          <div
            key={message.id}
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
                  {message.role === 'user' ? (
                    message.content
                  ): (
                    <div className="markdown-content prose prose-sm max-w-none">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                    </div>
                  )}
              </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="max-w-[80%] p-3 rounded-lg bg-white border border-gray-200">
              <div className="flex space-x-2">
                <div className="h-2 w-2 bg-gray-300 rounded-full animate-bounce"></div>
                <div className="h-2 w-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                <div className="h-2 w-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messageEndRef} />
      </div>
      {/* chat input */}
      <div className="border-t border-gray-200 p-4">
        <form
          onSubmit={handleSubmit}
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
              disabled={!inputMessage.trim() || isLoading}
              className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send size={20}/>
            </button>
        </form>
      </div>
    </div>
  )
}
// app/components/ChatInterface.tsx
import { ChevronDown, Mic, Send, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
}

const AVAILABLE_MODELS = [
  { id: "gpt-4", name: "GPT-4", description: "Most powerful model, but slower" },
  { id: "gpt-3.5-turbo", name: "GPT-3.5 Turbo", description: "Fast, good for most queries" },
  { id: "gpt-4-turbo", name: "GPT-4 Turbo", description: "Powerful with larger context" },
  { id: "gpt-4o", name: "GPT-4o", description: "Newest model with optimal performance" },
]

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSendMessage: (message: string, model: string) => Promise<void>;
  onVoiceRecord: () => void;
  isConversationSelected: boolean;
}

export default function ChatInterface({
  messages,
  onSendMessage,
  onVoiceRecord,
  isConversationSelected
}: ChatInterfaceProps) {
  // const [isRecording, setIsRecording] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const errorTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const [selectedModel, setSelectedModel] = useState(AVAILABLE_MODELS[0].id);

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
      await onSendMessage(inputMessage, selectedModel);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to send message. Please try again.';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }

  const toggleModelMenu = () => {
    setIsModelMenuOpen(!isModelMenuOpen);
  }

  const selectModel = (modelId: string) => {
    setIsModelMenuOpen(false);
    setSelectedModel(modelId);
  }

  // const handleVoiceRecord = () => {
  //   try {
  //     setIsRecording(!isRecording);
  //     onVoiceRecord();
  //   } catch (error) {
  //     const errorMessage = error instanceof Error ? error.message : 'Failed to start voice recording. Please try again.';
  //     setError(errorMessage);
  //   }
  // }

  const getSelectedModelName = () => {
    const model = AVAILABLE_MODELS.find(m => m.id === selectedModel);
    return model ? model.name : 'Select Model';
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
      <div className="border-t border-gray-200 p-2">
        <div className="relative mb-1" ref={modelMenuRef}>
          <button
            onClick={toggleModelMenu}
            className="inline-flex items-center text-sm text-gray-600 bg-gray-100 hover:bg-gray-200 px-3 py-1 rounded-md"
          >
            <span className="mr-1">Model: {getSelectedModelName()}</span>
            <ChevronDown size={16}/>
          </button>
          {isModelMenuOpen && (
            <div className="absolute bottom-full mb-2 left-0 bg-white shadow-lg rounded-md border border-gray-200 w-64 z-10">
              <div className="p-2">
                <h3 className="text-sm font-medium text-gray-700 mb-2">Select Model</h3>
                <div className="space-y-1">
                  {AVAILABLE_MODELS.map(model => (
                    <button
                      key={model.id}
                      onClick={() => selectModel(model.id)}
                      className={`w-full text-left px-3 py-2 text-sm rounded-md ${selectedModel === model.id ? 'bg-blue-100 text-blue-700' : 'hover:bg-gray-100'}`}
                    >
                      <div className="font-medium">{model.name}</div>
                      <div className="text-xs text-gray-500">{model.description}</div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
        <form
          onSubmit={handleSubmit}
          className="flex gap-2">
          {/* <button 
            type="button"
            onClick={handleVoiceRecord}
            className={`p-2 rounded-full ${
              isRecording ? 'bg-red-500' : 'bg-gray-200'
            } hover:opacity-80`}>
              <Mic size={20} className={isRecording ? 'text-white' : 'text-gray-600'}/>
          </button> */}
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
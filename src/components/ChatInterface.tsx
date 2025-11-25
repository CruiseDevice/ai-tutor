// app/components/ChatInterface.tsx
import { Bot, ChevronDown, Loader2, Send, Sparkles, User, Paperclip, StopCircle } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { userApi } from "@/lib/api-client";

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
  // onVoiceRecord,
  isConversationSelected
}: ChatInterfaceProps) {
  const router = useRouter();
  // const [isRecording, setIsRecording] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const errorTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const [selectedModel, setSelectedModel] = useState(AVAILABLE_MODELS[0].id);
  const [hasApiKey, setHasApiKey] = useState(false);

  const messageEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const checkApiKey = async () => {
      try{
        const response = await userApi.checkAPIKey();

        if(!response.ok) {
          throw new Error('Failed to check API key status');
        }

        const data = await response.json();
        setHasApiKey(data.hasApiKey);
      } catch (error) {
        console.error('Error checking API key:', error);
        setHasApiKey(false);
      }
    };
    checkApiKey();
  }, []);

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

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 150) + 'px';
    }
  }, [inputMessage]);

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputMessage.trim() || isLoading || !isConversationSelected) return;

    if (!hasApiKey) {
      setError('Please set up your OpenAI API key in settings to continue chatting');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      setInputMessage('');
      // Reset height
      if (textareaRef.current) textareaRef.current.style.height = 'auto';

      await onSendMessage(inputMessage, selectedModel);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to send message. Please try again.';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const toggleModelMenu = () => {
    setIsModelMenuOpen(!isModelMenuOpen);
  }

  const selectModel = (modelId: string) => {
    setIsModelMenuOpen(false);
    setSelectedModel(modelId);
  }

  const getSelectedModelName = () => {
    const model = AVAILABLE_MODELS.find(m => m.id === selectedModel);
    return model ? model.name : 'Select Model';
  }

  // Click outside to close model menu
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (modelMenuRef.current && !modelMenuRef.current.contains(event.target as Node)) {
        setIsModelMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [modelMenuRef]);

  return (
    <div className="w-1/2 h-full flex flex-col bg-slate-50/50 relative overflow-hidden">
       {/* Background Pattern */}
       <div className="absolute inset-0 z-0 opacity-[0.03]"
           style={{
             backgroundImage: 'radial-gradient(#64748b 1px, transparent 1px)',
             backgroundSize: '24px 24px'
           }}
      />

      {/* Error Toast */}
      {error && (
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-50 w-[90%] max-w-md animate-in fade-in slide-in-from-top-2">
          <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-xl shadow-lg flex items-start gap-3">
            <div className="p-1 bg-red-100 rounded-full">
              <StopCircle size={16} className="text-red-600"/>
            </div>
            <div className="flex-1 text-sm">
              <p className="font-medium">{error}</p>
              {!hasApiKey && (
                <button
                  onClick={() => router.push('/settings')}
                  className="mt-2 text-xs bg-red-100 hover:bg-red-200 text-red-700 px-3 py-1.5 rounded-md transition-colors font-medium"
                >
                  Go to Settings
                </button>
              )}
            </div>
            <button
              onClick={() => setError(null)}
              className="text-red-400 hover:text-red-600"
            >
              <span className="sr-only">Close</span>
              ×
            </button>
          </div>
        </div>
      )}

      {/* Header / Model Selector */}
      <div className="flex-shrink-0 px-6 py-4 border-b border-gray-100 bg-white/80 backdrop-blur-sm z-10 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white shadow-md shadow-blue-100">
            <Bot size={18} />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-800">AI Tutor</h2>
            <p className="text-xs text-slate-500 font-medium flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
              Online
            </p>
          </div>
        </div>

        <div className="relative" ref={modelMenuRef}>
          <button
            onClick={toggleModelMenu}
            className="flex items-center gap-2 text-xs font-medium text-slate-600 bg-slate-100 hover:bg-slate-200 px-3 py-1.5 rounded-full transition-colors border border-slate-200"
          >
            <Sparkles size={12} className="text-indigo-500"/>
            <span>{getSelectedModelName()}</span>
            <ChevronDown size={12} className={`transition-transform duration-200 ${isModelMenuOpen ? 'rotate-180' : ''}`}/>
          </button>

          {isModelMenuOpen && (
            <div className="absolute top-full right-0 mt-2 w-64 bg-white rounded-xl shadow-xl border border-gray-100 p-1.5 z-20 animate-in fade-in zoom-in-95 duration-200">
              <div className="px-2 py-1.5 mb-1">
                <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Select Model</h3>
              </div>
              {AVAILABLE_MODELS.map(model => (
                <button
                  key={model.id}
                  onClick={() => selectModel(model.id)}
                  className={`w-full text-left px-3 py-2.5 rounded-lg flex flex-col gap-0.5 transition-all ${
                    selectedModel === model.id
                      ? 'bg-blue-50 text-blue-700 ring-1 ring-blue-100'
                      : 'hover:bg-slate-50 text-slate-700'
                  }`}
                >
                  <span className="text-sm font-medium flex items-center gap-2">
                    {model.name}
                    {selectedModel === model.id && <span className="w-1.5 h-1.5 rounded-full bg-blue-500"></span>}
                  </span>
                  <span className="text-xs text-slate-400 line-clamp-1">{model.description}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Chat Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6 scrollbar-thin scrollbar-thumb-slate-200 scrollbar-track-transparent">
        {messages.length === 0 && !isConversationSelected && (
          <div className="flex flex-col items-center justify-center h-full text-center px-6 opacity-0 animate-in fade-in slide-in-from-bottom-4 duration-700">
            <div className="w-20 h-20 bg-blue-50 rounded-3xl flex items-center justify-center mb-6 shadow-sm transform rotate-3">
              <Paperclip size={32} className="text-blue-400" />
            </div>
            <h3 className="text-xl font-bold text-slate-800 mb-2">No Document Selected</h3>
            <p className="text-slate-500 max-w-xs text-sm leading-relaxed">
              Select a conversation from the sidebar or upload a new document to start chatting.
            </p>
          </div>
        )}

        {messages.length === 0 && isConversationSelected && (
          <div className="flex flex-col items-center justify-center h-full text-center px-6 opacity-0 animate-in fade-in slide-in-from-bottom-4 duration-700">
            <div className="w-20 h-20 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-3xl flex items-center justify-center mb-6 shadow-lg shadow-indigo-200 transform -rotate-3">
              <Bot size={40} className="text-white" />
            </div>
            <h3 className="text-xl font-bold text-slate-800 mb-2">How can I help?</h3>
            <p className="text-slate-500 max-w-xs text-sm leading-relaxed mb-8">
              Ask me anything about your document. I can summarize, explain concepts, or find specific details.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-md">
              {['Summarize this document', 'What are the key points?', 'Explain the methodology', 'List the main conclusions'].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => {
                    setInputMessage(suggestion);
                    if (textareaRef.current) textareaRef.current.focus();
                  }}
                  className="text-xs text-slate-600 bg-white border border-slate-200 hover:border-blue-300 hover:bg-blue-50 hover:text-blue-600 px-4 py-3 rounded-xl transition-all text-left shadow-sm"
                >
                  &quot;{suggestion}&quot;
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex gap-4 ${
              message.role==='user' ? 'justify-end' : 'justify-start'
            } animate-in fade-in slide-in-from-bottom-2 duration-300`}
          >
            {/* Avatar for Assistant */}
            {message.role !== 'user' && (
              <div className="w-8 h-8 rounded-lg bg-white border border-gray-100 flex-shrink-0 flex items-center justify-center shadow-sm mt-1">
                <Bot size={16} className="text-indigo-600" />
              </div>
            )}

            <div
              className={`max-w-[85%] sm:max-w-[75%] p-4 rounded-2xl shadow-sm text-sm leading-relaxed ${
                message.role === 'user'
                ? 'bg-gradient-to-br from-blue-600 to-blue-700 text-white rounded-tr-sm'
                : 'bg-white border border-gray-100 text-slate-700 rounded-tl-sm'
              }`}
            >
              {message.role === 'user' ? (
                message.content
              ) : (
                <div className="markdown-content prose prose-sm max-w-none prose-headings:text-slate-800 prose-p:text-slate-700 prose-a:text-blue-600 hover:prose-a:underline prose-strong:text-slate-900 prose-code:text-pink-600 prose-code:bg-pink-50 prose-code:px-1 prose-code:rounded prose-code:before:content-none prose-code:after:content-none prose-pre:bg-slate-900 prose-pre:text-slate-50">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
                </div>
              )}
            </div>

            {/* Avatar for User */}
            {message.role === 'user' && (
              <div className="w-8 h-8 rounded-lg bg-blue-100 flex-shrink-0 flex items-center justify-center mt-1">
                <User size={16} className="text-blue-600" />
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="flex gap-4 justify-start animate-in fade-in slide-in-from-bottom-2">
             <div className="w-8 h-8 rounded-lg bg-white border border-gray-100 flex-shrink-0 flex items-center justify-center shadow-sm mt-1">
                <Bot size={16} className="text-indigo-600" />
              </div>
            <div className="bg-white border border-gray-100 px-5 py-4 rounded-2xl rounded-tl-sm shadow-sm flex items-center gap-2">
              <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </div>
        )}
        <div ref={messageEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 bg-white/80 backdrop-blur-md border-t border-gray-100 relative z-10">
        <form
          onSubmit={handleSubmit}
          className="max-w-4xl mx-auto relative flex gap-3 items-end"
        >
          <div className="relative flex-1 bg-white border border-gray-200 rounded-2xl shadow-sm focus-within:ring-2 focus-within:ring-blue-100 focus-within:border-blue-400 transition-all overflow-hidden">
            <textarea
              ref={textareaRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question..."
              rows={1}
              className="w-full max-h-[150px] py-3 px-4 bg-transparent border-none focus:ring-0 resize-none text-sm text-slate-800 placeholder:text-slate-400"
              style={{ minHeight: '44px' }}
            />

            {/* Optional: Add attachment button or similar features here later */}
          </div>

          <button
            type="submit"
            disabled={!inputMessage.trim() || isLoading}
            className={`p-3 rounded-xl shadow-md flex items-center justify-center transition-all duration-200 ${
              !inputMessage.trim() || isLoading
                ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700 hover:shadow-lg hover:scale-105 active:scale-95'
            }`}
          >
            {isLoading ? (
              <Loader2 size={20} className="animate-spin" />
            ) : (
              <Send size={20} className={inputMessage.trim() ? 'ml-0.5' : ''} />
            )}
          </button>
        </form>
        <div className="text-center mt-2">
           <p className="text-[10px] text-slate-400">
             AI can make mistakes. Please review important information.
           </p>
        </div>
      </div>
    </div>
  )
}
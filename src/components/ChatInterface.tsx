// app/components/ChatInterface.tsx
import { Bot, ChevronDown, Loader2, Send, Sparkles, User, Paperclip, StopCircle, Search, FileText, ArrowRight, Copy, Check } from "lucide-react";
import { useRouter } from "next/navigation";
import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import SyntaxHighlighter from 'react-syntax-highlighter/dist/esm/prism';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { userApi } from "@/lib/api-client";
import type { AnnotationReference, AgentMetadata } from "@/types/annotations";
import AgentWorkflowProgress from "./AgentWorkflowProgress";

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  annotations?: AnnotationReference[];
  metadata?: AgentMetadata;
}

interface WorkflowStep {
  node: string;
  status: 'pending' | 'in_progress' | 'completed';
  data?: Record<string, unknown>;
}

const AVAILABLE_MODELS = [
  // GPT-5 Series (Chat Models)
  { id: "gpt-5.1", name: "GPT-5.1", description: "Latest GPT-5 model • $1.25/$0.125 per 1M tokens" },
  { id: "gpt-5", name: "GPT-5", description: "GPT-5 base model • $1.25/$0.125 per 1M tokens" },
  { id: "gpt-5-mini", name: "GPT-5 Mini", description: "Compact GPT-5 • $0.25/$0.025 per 1M tokens" },
  { id: "gpt-5-nano", name: "GPT-5 Nano", description: "Ultra-lightweight • $0.05/$0.005 per 1M tokens" },
  { id: "gpt-5.1-chat-latest", name: "GPT-5.1 Chat Latest", description: "Latest chat variant • $1.25/$0.125 per 1M tokens" },
  { id: "gpt-5-chat-latest", name: "GPT-5 Chat Latest", description: "Chat optimized • $1.25/$0.125 per 1M tokens" },
  { id: "gpt-5-pro", name: "GPT-5 Pro", description: "Premium performance • $15.00/$120.00 per 1M tokens" },

  // GPT-4.1 Series (Chat Models)
  { id: "gpt-4.1", name: "GPT-4.1", description: "Enhanced GPT-4 • $2.00/$0.50 per 1M tokens" },
  { id: "gpt-4.1-mini", name: "GPT-4.1 Mini", description: "Compact GPT-4.1 • $0.40/$0.10 per 1M tokens" },
  { id: "gpt-4.1-nano", name: "GPT-4.1 Nano", description: "Ultra-light GPT-4.1 • $0.10/$0.025 per 1M tokens" },

  // GPT-4o Series (Chat Models)
  { id: "gpt-4o", name: "GPT-4o", description: "Optimized GPT-4 • $2.50/$1.25 per 1M tokens" },
  { id: "gpt-4o-2024-05-13", name: "GPT-4o (2024-05-13)", description: "Snapshot version • $5.00/$15.00 per 1M tokens" },
  { id: "gpt-4o-mini", name: "GPT-4o Mini", description: "Lightweight GPT-4o • $0.15/$0.075 per 1M tokens" },

  // Realtime Series (Chat Models)
  { id: "gpt-realtime", name: "GPT Realtime", description: "Real-time responses • $4.00/$0.40 per 1M tokens" },
  { id: "gpt-realtime-mini", name: "GPT Realtime Mini", description: "Lightweight realtime • $0.60/$0.06 per 1M tokens" },
  { id: "gpt-4o-realtime-preview", name: "GPT-4o Realtime Preview", description: "Preview realtime • $5.00/$2.50 per 1M tokens" },
  { id: "gpt-4o-mini-realtime-preview", name: "GPT-4o Mini Realtime", description: "Mini realtime preview • $0.60/$0.30 per 1M tokens" },

  // O-Series (Reasoning Models - Chat Compatible)
  { id: "o1", name: "O1", description: "Reasoning model • $15.00/$7.50 per 1M tokens" },
  { id: "o1-pro", name: "O1 Pro", description: "Advanced reasoning • $150.00/$600.00 per 1M tokens" },
  { id: "o1-mini", name: "O1 Mini", description: "Lightweight reasoning • $1.10/$0.55 per 1M tokens" },
  { id: "o3", name: "O3", description: "Next-gen reasoning • $2.00/$0.50 per 1M tokens" },
  { id: "o3-pro", name: "O3 Pro", description: "Premium reasoning • $20.00/$80.00 per 1M tokens" },
  { id: "o3-mini", name: "O3 Mini", description: "Compact reasoning • $1.10/$0.55 per 1M tokens" },
  { id: "o3-deep-research", name: "O3 Deep Research", description: "Deep research mode • $10.00/$2.50 per 1M tokens" },
  { id: "o4-mini", name: "O4 Mini", description: "Latest mini reasoning • $1.10/$0.275 per 1M tokens" },
  { id: "o4-mini-deep-research", name: "O4 Mini Deep Research", description: "Mini deep research • $2.00/$0.50 per 1M tokens" },
]

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSendMessage: (message: string, model: string, useAgent: boolean) => Promise<void>;
  onVoiceRecord: () => void;
  isConversationSelected: boolean;
  onAnnotationClick?: (annotation: AnnotationReference) => void;
  workflowSteps?: WorkflowStep[];
  showWorkflow?: boolean;
}

// Helper function to convert LaTeX delimiters to markdown math format
const preprocessMathContent = (content: string): string => {
  // Split content into parts, separating code blocks and inline code
  const parts: Array<{ type: 'code' | 'text', content: string }> = [];

  // Match code blocks (```...```) and inline code (`...`)
  const codeBlockRegex = /(```[\s\S]*?```|`[^`\n]+?`)/g;
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    // Add text before code block
    if (match.index > lastIndex) {
      parts.push({ type: 'text', content: content.slice(lastIndex, match.index) });
    }
    // Add code block
    parts.push({ type: 'code', content: match[0] });
    lastIndex = match.index + match[0].length;
  }

  // Add remaining text
  if (lastIndex < content.length) {
    parts.push({ type: 'text', content: content.slice(lastIndex) });
  }

  // Process only text parts, leave code blocks unchanged
  return parts.map(part => {
    if (part.type === 'code') {
      return part.content;
    }

    let processed = part.content;

    // Convert \[ ... \] to $$ ... $$ for display math
    // Use a function replacer to avoid `$1` substitution quirks
    processed = processed.replace(/\\\[([\s\S]*?)\\\]/g, (_match, inner) => `$$${inner}$$`);

    // Convert \( ... \) to $ ... $ for inline math
    processed = processed.replace(/\\\((.*?)\\\)/g, (_match, inner) => `$${inner}$`);

    return processed;
  }).join('');
};

// Code block component with syntax highlighting and copy button
interface CodeBlockProps {
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
  [key: string]: unknown;
}

const CodeBlock = ({ inline, className, children }: CodeBlockProps) => {
  const [copied, setCopied] = useState(false);
  const match = /language-(\w+)/.exec(className || '');
  const language = match ? match[1] : '';

  // Extract code string from children - ensure it's always a plain string
  // ReactMarkdown passes children as an array of strings for code blocks
  // Use useMemo to ensure we don't recreate the string unnecessarily
  const codeString = useMemo(() => {
    const extracted = React.Children.toArray(children)
      .map((child) => {
        // For code blocks, children should be strings
        if (typeof child === 'string') {
          return child;
        }
        // If it's a React element, try to extract text content
        if (React.isValidElement(child)) {
          const element = child as React.ReactElement<{ children?: React.ReactNode }>;
          if (element.props?.children) {
            return React.Children.toArray(element.props.children)
              .map(c => typeof c === 'string' ? c : String(c))
              .join('');
          }
        }
        // Fallback: convert to string
        return String(child);
      })
      .join('')
      .replace(/\n$/, '');
    return extracted;
  }, [children]);

  // Determine if this is a code block (has language class and not inline)
  const hasLanguage = className && className.includes('language-');
  const isCodeBlock = hasLanguage && inline !== true;

  const handleCopy = async () => {
    await navigator.clipboard.writeText(codeString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Inline code - render as inline if explicitly inline or no language class
  if (!isCodeBlock || inline) {
    return (
      <code className="px-1.5 py-0.5 bg-pink-50 text-pink-600 rounded text-sm font-mono">
        {children}
      </code>
    );
  }

  // Code block with syntax highlighting
  // Render SyntaxHighlighter directly - it will replace the pre+code structure
  return (
    <div className="relative group my-4 -mx-4 sm:mx-0 overflow-hidden">
      <div className="absolute top-3 right-3 z-10">
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2.5 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs rounded-md transition-colors opacity-0 group-hover:opacity-100 shadow-lg"
          title="Copy code"
        >
          {copied ? (
            <>
              <Check size={14} />
              <span>Copied!</span>
            </>
          ) : (
            <>
              <Copy size={14} />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <div className="overflow-x-auto">
        <SyntaxHighlighter
          style={vscDarkPlus}
          language={language || 'text'}
          customStyle={{
            margin: 0,
            borderRadius: '0.5rem',
            padding: '1rem',
            paddingTop: '1.5rem',
            fontSize: '0.875rem',
            lineHeight: '1.5',
            overflow: 'visible',
          }}
          codeTagProps={{
            style: {
              fontSize: '0.875rem',
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
            }
          }}
        >
          {codeString}
        </SyntaxHighlighter>
      </div>
    </div>
  );
};

export default function ChatInterface({
  messages,
  onSendMessage,
  // onVoiceRecord,
  isConversationSelected,
  onAnnotationClick,
  workflowSteps = [],
  showWorkflow = false,
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
  const [modelSearchQuery, setModelSearchQuery] = useState('');
  const [useAgent, setUseAgent] = useState(true);

  const messageEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);
  const prevMessagesLengthRef = useRef(messages.length);

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

  // Track scroll position to determine if we should auto-scroll
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      const threshold = 150;
      const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < threshold;
      shouldAutoScrollRef.current = isNearBottom;
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    // Check initial position
    handleScroll();

    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  // scroll to bottom when messages change, but only if user is near bottom or new message was added
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return;

    const messagesIncreased = messages.length > prevMessagesLengthRef.current;
    const wasEmpty = prevMessagesLengthRef.current === 0;
    prevMessagesLengthRef.current = messages.length;

    // Only auto-scroll if new messages were added
    if (!messagesIncreased) return;

    // Auto-scroll if:
    // 1. User is near bottom (within 150px), OR
    // 2. This is the initial load (was empty before)
    if (shouldAutoScrollRef.current || wasEmpty) {
      // Use requestAnimationFrame to ensure DOM is updated, then scroll container directly
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          if (container) {
            container.scrollTop = container.scrollHeight;
          }
        });
      });
    }
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

    const messageToSend = inputMessage;
    setInputMessage('');
    // Reset height
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    setIsLoading(true);
    setError(null);

    try {
      // onSendMessage now handles streaming and manages its own loading state
      // But we keep isLoading here for UI feedback during streaming
      await onSendMessage(messageToSend, selectedModel, useAgent);
      // Loading will be cleared by Dashboard when streaming completes
      // But we'll also clear it here as a fallback after a delay
      setTimeout(() => setIsLoading(false), 100);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to send message. Please try again.';
      setError(errorMessage);
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
    if (!isModelMenuOpen) {
      setModelSearchQuery(''); // Reset search when opening
    }
  }

  const selectModel = (modelId: string) => {
    setIsModelMenuOpen(false);
    setSelectedModel(modelId);
  }

  const getSelectedModelName = () => {
    const model = AVAILABLE_MODELS.find(m => m.id === selectedModel);
    return model ? model.name : 'Select Model';
  }

  const filteredModels = AVAILABLE_MODELS.filter(model =>
    model.name.toLowerCase().includes(modelSearchQuery.toLowerCase()) ||
    model.id.toLowerCase().includes(modelSearchQuery.toLowerCase()) ||
    model.description.toLowerCase().includes(modelSearchQuery.toLowerCase())
  );

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
    <div className="w-full h-full flex flex-col bg-slate-50/50 relative overflow-hidden">
       {/* Background Pattern */}
       <div className="absolute inset-0 z-0 opacity-[0.03] pointer-events-none"
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
                  className="no-select no-tap-highlight mt-2 text-xs bg-red-100 hover:bg-red-200 text-red-700 px-3 py-1.5 rounded-md transition-colors font-medium min-h-[44px]"
                >
                  Go to Settings
                </button>
              )}
            </div>
            <button
              onClick={() => setError(null)}
              className="no-tap-highlight text-red-400 hover:text-red-600 min-w-[44px] min-h-[44px] flex items-center justify-center"
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

        <div className="flex items-center gap-4">
          <div className="relative" ref={modelMenuRef}>
            <button
              onClick={toggleModelMenu}
              className="no-select no-tap-highlight flex items-center gap-2 text-xs font-medium text-slate-600 bg-slate-100 hover:bg-slate-200 px-3 py-1.5 rounded-full transition-colors border border-slate-200 min-h-[44px]"
            >
              <Sparkles size={12} className="text-indigo-500"/>
              <span>{getSelectedModelName()}</span>
              <ChevronDown size={12} className={`transition-transform duration-200 ${isModelMenuOpen ? 'rotate-180' : ''}`}/>
            </button>

            {isModelMenuOpen && (
            <div className="absolute top-full right-0 mt-2 w-80 bg-white rounded-xl shadow-xl border border-gray-100 z-20 animate-in fade-in zoom-in-95 duration-200 overflow-hidden flex flex-col max-h-[600px]">
              <div className="px-3 py-2 border-b border-gray-100 flex-shrink-0">
                <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Select Model</h3>
                <div className="relative">
                  <Search size={14} className="absolute left-2.5 top-1/2 transform -translate-y-1/2 text-slate-400" />
                  <input
                    type="text"
                    placeholder="Search models..."
                    value={modelSearchQuery}
                    onChange={(e) => setModelSearchQuery(e.target.value)}
                    onClick={(e) => e.stopPropagation()}
                    className="w-full pl-8 pr-3 py-1.5 text-xs border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 text-slate-700 placeholder:text-slate-400"
                  />
                </div>
              </div>
              <div className="overflow-y-auto p-1.5 flex-1">
                {filteredModels.length === 0 ? (
                  <div className="px-3 py-8 text-center text-sm text-slate-400">
                    No models found matching &quot;{modelSearchQuery}&quot;
                  </div>
                ) : (
                  filteredModels.map(model => (
                    <button
                      key={model.id}
                      onClick={() => selectModel(model.id)}
                      className={`w-full text-left px-3 py-2.5 rounded-lg flex flex-col gap-0.5 transition-all mb-1 ${
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
                  ))
                )}
              </div>
            </div>
            )}
          </div>

          {/* Agent Mode Toggle */}
          <div className="flex items-center gap-2">
            <label className="no-tap-highlight relative inline-flex items-center cursor-pointer min-h-[44px]">
              <input
                type="checkbox"
                checked={useAgent}
                onChange={(e) => setUseAgent(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-1/2 after:-translate-y-1/2 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
            </label>
            <div className="flex flex-col">
              <span className="text-xs font-medium text-gray-700">
                Agent Mode
              </span>
              {useAgent && (
                <span className="text-[10px] text-blue-600">Advanced reasoning</span>
              )}
            </div>
          </div>

          {/* Info Tooltip */}
          <div className="relative group">
            <svg className="w-4 h-4 text-gray-400 cursor-help" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-10 w-64">
              Agent mode uses multi-step reasoning for better quality answers
              <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-gray-900"></div>
            </div>
          </div>
        </div>
      </div>

      {/* Chat Messages Area */}
      <div
        ref={messagesContainerRef}
        onClick={() => {
          // Dismiss keyboard when tapping outside the input area on touch devices
          if (textareaRef.current && document.activeElement === textareaRef.current) {
            textareaRef.current.blur();
          }
        }}
        className="flex-1 min-h-0 overflow-y-auto p-4 sm:p-6 space-y-6 scrollbar-thin scrollbar-thumb-slate-200 scrollbar-track-transparent"
      >
        {/* Workflow Progress */}
        <AgentWorkflowProgress steps={workflowSteps} visible={showWorkflow} />

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
                  className="no-select no-tap-highlight text-xs text-slate-600 bg-white border border-slate-200 hover:border-blue-300 hover:bg-blue-50 hover:text-blue-600 px-4 py-3 rounded-xl transition-all text-left shadow-sm min-h-[44px]"
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
              className={`max-w-[85%] sm:max-w-[75%] rounded-2xl shadow-sm text-sm leading-relaxed ${
                message.role === 'user'
                ? 'bg-gradient-to-br from-blue-600 to-blue-700 text-white rounded-tr-sm p-4'
                : 'bg-white border border-gray-100 text-slate-700 rounded-tl-sm overflow-visible'
              }`}
            >
              {message.role === 'user' ? (
                message.content
              ) : (
                <>
                  <div className="p-4 markdown-content prose prose-sm max-w-none prose-headings:text-slate-800 prose-p:text-slate-700 prose-a:text-blue-600 hover:prose-a:underline prose-strong:text-slate-900 prose-code:text-pink-600 prose-code:bg-pink-50 prose-code:px-1 prose-code:rounded prose-code:before:content-none prose-code:after:content-none prose-pre:bg-transparent prose-pre:p-0 prose-pre:m-0 prose-pre:border-0 overflow-x-auto">
                    <ReactMarkdown
                      remarkPlugins={[
                        remarkGfm,
                        [remarkMath, { singleDollarTextMath: true }]
                      ]}
                      rehypePlugins={[
                        [rehypeKatex, {
                          strict: false,
                          trust: true,
                          fleqn: false
                        }]
                      ]}
                      components={{
                        code: CodeBlock,
                        pre: ({ children }: { children?: React.ReactNode }) => {
                          // ReactMarkdown wraps code blocks in <pre><code>
                          // CodeBlock component will render SyntaxHighlighter which replaces both pre and code
                          // So we just pass through the code element - CodeBlock handles the rendering
                          return <>{children}</>;
                        },
                      }}
                    >
                      {preprocessMathContent(message.content)}
                    </ReactMarkdown>
                  </div>

                  {/* Annotation References */}
                  {message.annotations && message.annotations.length > 0 && (
                    <div className="border-t border-gray-100 bg-gradient-to-r from-yellow-50 to-amber-50 p-3">
                      <div className="flex items-center gap-2 text-xs font-medium text-amber-700 mb-2">
                        <FileText size={14} />
                        <span>Referenced in PDF</span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {message.annotations.map((annotation, idx) => (
                          <button
                            key={`${message.id}-annotation-${idx}`}
                            onClick={() => onAnnotationClick?.(annotation)}
                            className="group flex items-center gap-2 px-3 py-1.5 bg-white border border-amber-200 rounded-lg text-xs text-slate-700 hover:bg-amber-100 hover:border-amber-300 hover:text-amber-800 transition-all shadow-sm hover:shadow"
                          >
                            {annotation.sourceImageUrl && (
                              <span className="h-6 w-6 rounded border border-amber-200 overflow-hidden bg-white flex-shrink-0">
                                <img
                                  src={annotation.sourceImageUrl}
                                  alt="Annotation preview"
                                  className="h-full w-full object-cover"
                                  loading="lazy"
                                />
                              </span>
                            )}
                            <span className="font-semibold text-amber-600">
                              Page {annotation.pageNumber}
                            </span>
                            {annotation.explanation && (
                              <>
                                <span className="text-gray-300">|</span>
                                <span className="max-w-[150px] truncate text-slate-500 group-hover:text-slate-700">
                                  {annotation.explanation}
                                </span>
                              </>
                            )}
                            <ArrowRight size={12} className="text-amber-500 group-hover:translate-x-0.5 transition-transform" />
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </>
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
            className={`no-select no-tap-highlight p-3 rounded-xl shadow-md flex items-center justify-center transition-all duration-200 min-w-[44px] min-h-[44px] ${
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

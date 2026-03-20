// app/components/ChatInterface.tsx
import { Loader2, StopCircle, Paperclip } from "lucide-react";
import { useRouter } from "next/navigation";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { userApi } from "@/lib/api-client";
import type { AnnotationReference } from "@/types/annotations";
import AgentWorkflowProgress from "./AgentWorkflowProgress";
import { ChatMessage } from "./Chat/ChatMessage";

// Store imports for Zustand migration
import { useChatStore, selectMessages, selectIsLoading } from '@/stores/chatStore';
import { useAnnotationsStore } from '@/stores/annotationsStore';

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

export default function ChatInterface() {
  const router = useRouter();

  // =====================================================
  // STORE HOOKS
  // =====================================================
  const messages = useChatStore(selectMessages);
  const isLoading = useChatStore(selectIsLoading);
  const conversationId = useChatStore((s) => s.conversationId);
  const selectedModel = useChatStore((s) => s.selectedModel);
  const sendMessage = useChatStore((s) => s.sendMessage);
  const setSelectedModel = useChatStore((s) => s.setSelectedModel);
  const setApiKeyStatus = useChatStore((s) => s.setApiKeyStatus);
  const hasApiKey = useChatStore((s) => s.hasApiKey);
  const useAgent = useChatStore((s) => s.useAgent);
  const toggleAgent = useChatStore((s) => s.toggleAgent);
  const workflowSteps = useChatStore((s) => s.workflowSteps);
  const showWorkflow = useChatStore((s) => s.showWorkflow);
  const setSelectedAnnotation = useAnnotationsStore((s) => s.setSelectedAnnotation);

  // Local state
  const [inputMessage, setInputMessage] = useState('');
  const [error, setError] = useState<string | null>(null);
  const errorTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const [modelSearchQuery, setModelSearchQuery] = useState('');

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
        setApiKeyStatus(data.hasApiKey);
      } catch (error) {
        console.error('Error checking API key:', error);
        setApiKeyStatus(false);
      }
    };
    checkApiKey();
  }, [setApiKeyStatus]);

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
    if (!inputMessage.trim() || isLoading || !conversationId) return;

    if (!hasApiKey) {
      setError('Please set up your OpenAI API key in settings to continue chatting');
      return;
    }

    const messageToSend = inputMessage;
    setInputMessage('');
    // Reset height
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    setError(null);

    try {
      await sendMessage(messageToSend);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to send message. Please try again.';
      setError(errorMessage);
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

  const handleToggleAgent = () => {
    toggleAgent();
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

  // Handler for annotation click
  const handleAnnotationClick = useCallback((annotation: AnnotationReference) => {
    setSelectedAnnotation(annotation.pageNumber.toString());
  }, [setSelectedAnnotation]);

  return (
    <div className="w-full h-full flex flex-col bg-panel-bg relative overflow-hidden">
      {/* =====================================================
          [004] HEADER - AI Tutor Panel
          ===================================================== */}
      <div className="flex-shrink-0 border-b-2 border-ink bg-panel-bg z-10">
        <div className="flex items-center justify-between px-4 py-3">
          {/* Left: Panel Number & AI Tutor Label */}
          <div className="flex items-center gap-3">
            <span className="font-mono text-xs text-accent">[004]</span>
            <div>
              <h2 className="font-mono font-bold text-sm uppercase">AI Tutor</h2>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs px-2 py-0.5 bg-accent text-paper">[ONLINE]</span>
              </div>
            </div>
          </div>

          {/* Right: Model Selector & Agent Mode */}
          <div className="flex items-center gap-4">
            {/* Model Selector */}
            <div className="relative" ref={modelMenuRef}>
              <button
                onClick={toggleModelMenu}
                className="no-select font-mono text-xs px-3 py-1.5 border border-ink hover:bg-ink hover:text-paper transition-colors min-h-[44px] flex items-center gap-2"
              >
                <span>[⚡]</span>
                <span>{getSelectedModelName()}</span>
                <span>[{isModelMenuOpen ? '▲' : '▼'}]</span>
              </button>

              {isModelMenuOpen && (
              <div className="absolute top-full right-0 mt-2 w-80 bg-panel-bg border-2 border-ink z-20 overflow-hidden flex flex-col max-h-[600px]">
                {/* Search Header */}
                <div className="px-3 py-2 border-b-2 border-ink flex-shrink-0">
                  <h3 className="font-mono text-xs text-accent uppercase tracking-wider mb-2">[SELECT MODEL]</h3>
                  <input
                    type="text"
                    placeholder="[Search models...]"
                    value={modelSearchQuery}
                    onChange={(e) => setModelSearchQuery(e.target.value)}
                    onClick={(e) => e.stopPropagation()}
                    className="w-full px-3 py-1.5 font-mono text-xs border border-ink bg-paper focus:outline-none focus:ring-2 focus:ring-accent text-ink placeholder:text-subtle"
                  />
                </div>

                {/* Model List */}
                <div className="overflow-y-auto p-1.5 flex-1 scrollbar-thin scrollbar-thumb-slate-200">
                  {filteredModels.length === 0 ? (
                    <div className="px-3 py-8 text-center font-mono text-xs text-subtle">
                      No models found matching &quot;{modelSearchQuery}&quot;
                    </div>
                  ) : (
                    filteredModels.map(model => (
                      <button
                        key={model.id}
                        onClick={() => selectModel(model.id)}
                        className={`w-full text-left px-3 py-2 border flex flex-col gap-0.5 transition-all mb-1 font-mono text-xs ${
                          selectedModel === model.id
                            ? 'bg-ink text-paper border-ink'
                            : 'bg-paper text-ink border-subtle hover:bg-ink hover:text-paper hover:border-ink'
                        }`}
                      >
                        <span className="font-bold flex items-center gap-2">
                          {model.name}
                          {selectedModel === model.id && <span>[✓]</span>}
                        </span>
                        <span className="text-subtle text-[10px]">{model.description}</span>
                      </button>
                    ))
                  )}
                </div>
              </div>
              )}
            </div>

            {/* Agent Mode Toggle */}
            <div className="flex items-center gap-2">
              <label className="no-tap-highlight flex items-center gap-2 cursor-pointer min-h-[44px] font-mono text-xs">
                <input
                  type="checkbox"
                  checked={useAgent}
                  onChange={handleToggleAgent}
                  className="accent-accent w-4 h-4"
                />
                <span>Agent Mode [{useAgent ? 'ON' : 'OFF'}]</span>
              </label>
            </div>
          </div>
        </div>
      </div>

      {/* =====================================================
          ERROR TOAST - Brutalist Style
          ===================================================== */}
      {error && (
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-50 w-[90%] max-w-md">
          <div className="bg-accent text-paper border-2 border-accent px-4 py-3 flex items-start gap-3">
            <span className="font-mono text-xl">[!]</span>
            <div className="flex-1 font-serif text-sm">
              <p className="font-bold">{error}</p>
              {!hasApiKey && (
                <button
                  onClick={() => router.push('/settings')}
                  className="brutalist-button brutalist-button-primary font-mono text-xs px-3 py-1.5 mt-2 min-h-[44px]"
                >
                  [GO TO SETTINGS]
                </button>
              )}
            </div>
            <button
              onClick={() => setError(null)}
              className="no-tap-highlight text-paper hover:underline min-w-[44px] min-h-[44px] flex items-center justify-center font-mono"
            >
              [×]
            </button>
          </div>
        </div>
      )}

      {/* =====================================================
          MESSAGES AREA
          ===================================================== */}
      <div
        ref={messagesContainerRef}
        onClick={() => {
          // Dismiss keyboard when tapping outside the input area on touch devices
          if (textareaRef.current && document.activeElement === textareaRef.current) {
            textareaRef.current.blur();
          }
        }}
        className="flex-1 min-h-0 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-slate-200 scrollbar-track-transparent"
      >
        {/* Workflow Progress */}
        <AgentWorkflowProgress key="workflow-progress" steps={workflowSteps} visible={showWorkflow} />

        {/* Empty State - No Document */}
        {messages.length === 0 && !!!conversationId && (
          <div key="no-document-state" className="flex flex-col items-center justify-center h-full text-center px-6">
            <div className="w-20 h-20 border-2 border-ink flex items-center justify-center mb-6">
              <Paperclip size={32} className="text-subtle" />
            </div>
            <h3 className="font-mono text-xl font-bold text-ink mb-2">[NO DOCUMENT SELECTED]</h3>
            <p className="font-serif text-subtle max-w-xs text-sm leading-relaxed">
              Select a conversation from the sidebar or upload a new document to start chatting.
            </p>
          </div>
        )}

        {/* Empty State - With Document */}
        {messages.length === 0 && !!conversationId && (
          <div key="empty-state" className="flex flex-col items-center justify-center h-full text-center px-6">
            <div className="w-20 h-20 bg-ink text-paper border-2 border-ink flex items-center justify-center mb-6 font-mono text-3xl">
              [AI]
            </div>
            <h3 className="font-mono text-xl font-bold text-ink mb-2">[HOW CAN I HELP?]</h3>
            <p className="font-serif text-subtle max-w-xs text-sm leading-relaxed mb-8">
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
                  className="no-select font-serif text-sm text-ink bg-paper border border-ink hover:bg-ink hover:text-paper px-4 py-3 transition-colors text-left min-h-[44px]"
                >
                  &quot;{suggestion}&quot;
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Messages */}
        {messages.map((message) => (
          <ChatMessage
            key={message.id}
            messageId={message.id}
            onAnnotationClick={handleAnnotationClick}
          />
        ))}

        {/* Loading Indicator */}
        {isLoading && (
          <div key="loading-indicator" className="flex gap-4 justify-start">
             <div className="w-8 h-8 border-2 border-ink flex-shrink-0 flex items-center justify-center font-mono text-xs bg-paper">
                [AI]
              </div>
            <div className="bg-paper border-2 border-ink px-4 py-3 flex items-center gap-2">
              <div className="w-2 h-2 bg-accent animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="w-2 h-2 bg-accent animate-bounce" style={{ animationDelay: '150ms' }} />
              <div className="w-2 h-2 bg-accent animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </div>
        )}
        <div key="message-end-ref" ref={messageEndRef} />
      </div>

      {/* =====================================================
          INPUT AREA - Brutalist Style
          ===================================================== */}
      <div className="border-t-2 border-ink bg-panel-bg p-4">
        <form
          onSubmit={handleSubmit}
          className="max-w-4xl mx-auto relative flex gap-3 items-end"
        >
          <div className="relative flex-1 bg-paper border-2 border-ink focus-within:ring-2 focus-within:ring-accent overflow-hidden">
            <textarea
              ref={textareaRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="[Ask a question about the document...]"
              rows={1}
              className="w-full max-h-[150px] py-3 px-4 bg-transparent border-none focus:ring-0 resize-none font-serif text-sm text-ink placeholder:text-subtle"
              style={{ minHeight: '44px' }}
            />
          </div>

          <button
            type="submit"
            disabled={!inputMessage.trim() || isLoading}
            className={`no-select font-mono text-sm px-4 py-3 min-w-[44px] min-h-[44px] flex items-center justify-center transition-all duration-150 ${
              !inputMessage.trim() || isLoading
                ? 'bg-paper text-subtle border border-ink cursor-not-allowed'
                : 'bg-ink text-paper border-2 border-ink hover:bg-accent hover:border-accent hover:text-paper'
            }`}
          >
            [{isLoading ? '...' : 'SEND'}]
          </button>
        </form>
        <div className="text-center mt-2">
           <p className="font-mono text-[10px] text-subtle">
             AI can make mistakes. Please review important information.
           </p>
        </div>
      </div>
    </div>
  )
}

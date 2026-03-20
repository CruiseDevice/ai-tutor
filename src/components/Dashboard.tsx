// app/components/Dashboard.tsx
"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react"
import EnhancedPDFViewer, { PDFViewerRef } from "./EnhancedPDFViewer";
import ChatInterface from "./ChatInterface";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import ChatSidebar, { ChatSidebarRef } from "./ChatSidebar";
import { authApi, documentApi, conversationApi, configApi, getPDFProxyUrl } from "@/lib/api-client";

// Store imports for Zustand migration
import { useChatStore } from '@/stores/chatStore';
import { useAnnotationsStore } from '@/stores/annotationsStore';
import { useAuthStore } from '@/stores/authStore';

function DashboardWithSearchParams () {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatSidebarRef = useRef<ChatSidebarRef>(null);
  const pdfViewerRef = useRef<PDFViewerRef>(null);

  // =====================================================
  // STORE HOOKS (PDF, annotations, and auth managed by stores)
  // =====================================================
  const storeSetCurrentPDF = useChatStore((s) => s.setCurrentPDF);
  const storeSetConversation = useChatStore((s) => s.setConversation);
  const storeSetMessages = useChatStore((s) => s.setMessages);
  const storeClearChat = useChatStore((s) => s.clearChat);
  const storeClearAnnotations = useAnnotationsStore((s) => s.clearAnnotations);
  const storeMessages = useChatStore((s) => s.messages);
  const storeAnnotations = useAnnotationsStore((s) => s.currentAnnotations);
  const storeConversationId = useChatStore((s) => s.conversationId);
  const storeIsLoading = useChatStore((s) => s.isLoading);
  const storeLoadConversation = useChatStore((s) => s.loadConversation);
  const storeMaxFileSize = useAuthStore((s) => s.maxFileSize);
  const storeSetMaxFileSize = useAuthStore((s) => s.setMaxFileSize);

  // =====================================================
  // AUTO-NAVIGATION: Watch for new annotations and navigate PDF
  // =====================================================
  const prevAnnotationsLengthRef = useRef(0);
  useEffect(() => {
    // Only navigate when annotations are ADDED (not cleared)
    if (storeAnnotations.length > 0 && storeAnnotations.length > prevAnnotationsLengthRef.current) {
      const firstAnnotation = storeAnnotations[0];
      if (pdfViewerRef.current && firstAnnotation) {
        console.log('[Dashboard] Auto-navigating to annotation:', firstAnnotation.pageNumber);
        pdfViewerRef.current.goToPage(firstAnnotation.pageNumber);

        const firstTextMatch = firstAnnotation.annotations?.find(
          (annotation: { textContent?: string }) => annotation.textContent
        )?.textContent || firstAnnotation.sourceText;
        if (firstTextMatch) {
          setTimeout(() => {
            pdfViewerRef.current?.highlightText(firstAnnotation.pageNumber, firstTextMatch);
          }, 500);
        }
      }
    }
    prevAnnotationsLengthRef.current = storeAnnotations.length;
  }, [storeAnnotations]);

  // =====================================================
  // SIDEBAR REFRESH: Trigger refresh after first message
  // =====================================================
  const prevMessagesLengthRef = useRef(0);
  useEffect(() => {
    const wasEmpty = prevMessagesLengthRef.current === 0;
    const nowHasMessages = storeMessages.length > 0;

    if (wasEmpty && nowHasMessages && chatSidebarRef.current) {
      // This was the first message, refresh sidebar to update conversation title
      setTimeout(() => {
        if (chatSidebarRef.current) {
          const sidebar = chatSidebarRef.current as unknown as { refreshConversations: () => void };
          if (typeof sidebar.refreshConversations === 'function') {
            sidebar.refreshConversations();
          }
        }
      }, 1000);
    }
    prevMessagesLengthRef.current = storeMessages.length;
  }, [storeMessages]);

  // Resizer state
  const [splitPosition, setSplitPosition] = useState(60); // Percentage (60% for PDF, 40% for Chat)
  const [isResizing, setIsResizing] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Function to update URL with the chat ID
  const updateUrl = useCallback((chatId: string | null) => {
    // Don't update URL if chatId is null, undefined or empty
    if (!chatId) {
      return;
    }
    // create a new URLSearchParams object
    const params = new URLSearchParams(searchParams.toString());

    // Set the chat parameter
    params.set('chat', chatId);

    // Update the URL without causing a page refresh
    router.push(`${pathname}?${params.toString()}`, {scroll: false});
  }, [searchParams, pathname, router]);

  // Sync store conversationId to URL (when conversation is selected in sidebar)
  useEffect(() => {
    if (storeConversationId) {
      updateUrl(storeConversationId);
    }
  }, [storeConversationId, updateUrl]);

  useEffect(() => {
    // debugging log
    const checkAuth = async () => {
      try {
        const response = await authApi.verifySession();

        if(!response.ok) {
          router.push('/login')
        }
      } catch (error) {
        // TODO: Display error message to user
        console.error('Auth check error: ', error);
        router.push('/login');
      }
    };
    checkAuth();
  }, [router])

  // fetch config (e.g., max file size) on component mount
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await configApi.get();
        if (response.ok) {
          const data = await response.json();
          storeSetMaxFileSize(data.max_file_size);
        } else {
          console.warn('Failed to fetch config, using default max file size');
        }
      } catch (error) {
        console.warn('Error fetching config, using default max file size:', error);
      }
    };
    fetchConfig();
  }, [storeSetMaxFileSize]);

  // Effect to restore chat from URL parameter (on page load or URL change)
  const isInitialMount = useRef(true);
  useEffect(() => {
    const chatId = searchParams.get('chat');

    // Skip clearing on initial render - let the app start fresh
    // Only clear if URL explicitly changes from having a chat to not having one
    if (isInitialMount.current) {
      isInitialMount.current = false;
      // Still try to restore if URL has chat param
      if (chatId && chatId !== storeConversationId && !storeIsLoading) {
        const restoreConversation = async () => {
          try {
            const data = await fetchConversations();
            const matchingConversation = data.find(
              (convo: {id: string}) => convo.id === chatId
            );
            if(matchingConversation) {
              await storeLoadConversation(chatId);
            }
          } catch (error) {
            console.error('Error restoring conversations: ', error);
            setError('Failed to restore conversation from URL');
          }
        };
        restoreConversation();
      }
      return;
    }

    // If URL has no chat param and we previously had one, clear the chat
    // This handles the case of navigating away from a conversation
    if (!chatId && storeConversationId) {
      storeClearChat();
      return;
    }

    // Restore conversation from URL if different from current
    if (chatId && chatId !== storeConversationId && !storeIsLoading) {
      const restoreConversation = async () => {
        try {
          const data = await fetchConversations();
          const matchingConversation = data.find(
            (convo: {id: string}) => convo.id === chatId
          );
          if(matchingConversation) {
            await storeLoadConversation(chatId);
          }
        } catch (error) {
          console.error('Error restoring conversations: ', error);
          setError('Failed to restore conversation from URL');
        }
      };
      restoreConversation();
    }
  }, [searchParams, storeConversationId, storeIsLoading, storeLoadConversation, storeClearChat]);

  const fetchConversations = async () => {
    try {
      const response = await conversationApi.list();
      if(!response.ok) {
        throw new Error('Failed to fetch conversations');
      }
      const data = await response.json();
      return data;
    } catch (error) {
      // TODO: Display error message to user
      console.error('Error fetching conversations: ', error);
      setError('Failed to fetch conversations');
      return [];
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
   // handlefileupload function triggered
    const file = e.target.files?.[0];

    if(!file) return;

    if(file.type !== 'application/pdf') {
      setError('Please upload a PDF file');
      return;
    }

    // Validate file size using config from backend (via authStore)
    if (file.size > storeMaxFileSize) {
      const maxSizeMB = (storeMaxFileSize / (1024 * 1024)).toFixed(0);
      setError(`File size should be less than ${maxSizeMB}MB`);
      return;
    }

    try{
      // upload file using document API
      const response = await documentApi.upload(file);

      if(!response.ok) {
        throw new Error('Failed to upload document');
      }

      const data = await response.json();
      storeSetCurrentPDF(getPDFProxyUrl(data.id));

      // reset messages and annotations for new document
      storeSetMessages([]);
      storeClearAnnotations();
      pdfViewerRef.current?.clearAnnotations();

      // set new conversation id from the response
      if(data.conversationId){
        storeSetConversation(data.conversationId, data.id, getPDFProxyUrl(data.id));
        // update URL with the new conversation ID
        updateUrl(data.conversationId);

        // a minimap conversation object to pass to the ChatSidebar
        // This prevents needing to wait for a separate fetch
        const newConversation = {
          id: data.conversationId,
          documentId: data.id,
          title: file.name,
          updatedAt: new Date().toISOString(),
        }

        // add it to the ChatSidebar component by passing it as a prop
        if (chatSidebarRef.current) {
          const chatSidebar = chatSidebarRef.current as unknown as { addNewConversation: (conv: typeof newConversation) => void };
          if (typeof chatSidebar.addNewConversation === 'function') {
            chatSidebar.addNewConversation(newConversation);
          } else {
            fetchConversations();
          }
        } else {
          fetchConversations();
        }
      } else {
        console.error('No conversationId returned from server');
      }

      // Process the document to generate embeddings
      try {
        const processResponse = await documentApi.process(data.id);
        if (!processResponse.ok) {
          const errorData = await processResponse.json().catch(() => ({ detail: 'Failed to process document' }));
          setError(`Document uploaded but processing failed: ${errorData.detail || errorData.error}`);
        } else {
          const processData = await processResponse.json();
          console.log('Document processed successfully:', processData);
        }
      } catch (processError) {
        console.error('Error processing document:', processError);
        setError('Document uploaded but failed to generate embeddings. Please try processing again.');
      }
    } catch (error) {
      // TODO: Display error message to user
      console.error('Upload error:', error);
      setError('Failed to upload document');
    }
  }

  const handleVoiceRecord = () => {
    // implement voice recording logic later
    console.log('Voice recording toggled');
  }

  // Resizer handlers - Pointer Events for touch + mouse support
  const handlePointerDown = (e: React.PointerEvent) => {
    // Only respond to primary touch/click (left mouse, first finger)
    if (!e.isPrimary) return;

    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    setIsResizing(true);
  };

  useEffect(() => {
    const handlePointerMove = (e: PointerEvent) => {
      if (!isResizing || !containerRef.current) return;

      const containerRect = containerRef.current.getBoundingClientRect();
      const newPosition = ((e.clientX - containerRect.left) / containerRect.width) * 100;

      // Constrain between 20% and 80% to prevent components from becoming too small
      const constrainedPosition = Math.max(20, Math.min(80, newPosition));
      setSplitPosition(constrainedPosition);
    };

    const handlePointerUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener('pointermove', handlePointerMove);
      document.addEventListener('pointerup', handlePointerUp);
      document.addEventListener('pointercancel', handlePointerUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.touchAction = 'none'; // Prevent scrolling while resizing on touch
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('pointermove', handlePointerMove);
      document.removeEventListener('pointerup', handlePointerUp);
      document.removeEventListener('pointercancel', handlePointerUp);
      document.body.style.cursor = '';
      document.body.style.touchAction = '';
      document.body.style.userSelect = '';
    };
  }, [isResizing]);

  if (error) {
    return <div className="p-4 text-red-500">{error}</div>
  }

  return (
    <div className="h-screen flex overflow-hidden">
      {/* Sidebar */}
      <ChatSidebar
        ref={chatSidebarRef}
      />

      {/* Main content Area */}
      <div ref={containerRef} className="flex flex-1 overflow-hidden relative">
        {/* PDF Viewer Section */}
        <div
          className="h-full overflow-hidden"
          style={{ width: `${splitPosition}%` }}
        >
          <EnhancedPDFViewer
            ref={pdfViewerRef}
            onFileUpload={handleFileUpload}
            fileInputRef={fileInputRef as React.RefObject<HTMLInputElement>}
          />
        </div>

        {/* Resizer */}
        <div
          onPointerDown={handlePointerDown}
          className={`no-select no-tap-highlight absolute top-0 bottom-0 w-1 bg-gray-200 hover:bg-blue-500 cursor-col-resize transition-colors z-20 ${
            isResizing ? 'bg-blue-500' : ''
          }`}
          style={{ left: `${splitPosition}%`, transform: 'translateX(-50%)' }}
        >
          <div className={`no-select no-tap-highlight absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-8 h-12 rounded-full flex items-center justify-center transition-all min-w-[44px] min-h-[44px] ${
            isResizing
              ? 'bg-blue-500 shadow-lg scale-110'
              : 'bg-gray-200 hover:bg-blue-400 hover:shadow-md'
          }`}>
            <div className="flex flex-col gap-1">
              <div className={`w-0.5 h-1 ${isResizing ? 'bg-white' : 'bg-gray-500'}`}></div>
              <div className={`w-0.5 h-1 ${isResizing ? 'bg-white' : 'bg-gray-500'}`}></div>
              <div className={`w-0.5 h-1 ${isResizing ? 'bg-white' : 'bg-gray-500'}`}></div>
            </div>
          </div>
        </div>

        {/* Chat Section */}
        <div
          className="h-full overflow-hidden"
          style={{ width: `${100 - splitPosition}%` }}
        >
          <ChatInterface
            onVoiceRecord={handleVoiceRecord}
          />
        </div>
      </div>
    </div>
  )
}

// Main export that uses Suspense
export default function Dashboard() {
  return (
    <Suspense fallback={<div className="h-screen flex items-center justify-center">Loading dashboard...</div>}>
      <DashboardWithSearchParams />
    </Suspense>
  )
}

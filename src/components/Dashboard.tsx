// app/components/Dashboard.tsx
"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react"
import EnhancedPDFViewer, { PDFViewerRef } from "./EnhancedPDFViewer";
import ChatInterface from "./ChatInterface";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import ChatSidebar, { ChatSidebarRef } from "./ChatSidebar";
import { authApi, documentApi, chatApi, conversationApi, configApi } from "@/lib/api-client";
import type { AnnotationReference } from "@/types/annotations";

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  annotations?: AnnotationReference[];
}

function DashboardWithSearchParams () {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const [currentPDF, setCurrentPDF] = useState('');
  const [documentId, setDocumentId] = useState('');
  const [userId, setUserId] = useState('');
  const [userEmail, setUserEmail] = useState('');
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentAnnotations, setCurrentAnnotations] = useState<AnnotationReference[]>([]);
  const [maxFileSize, setMaxFileSize] = useState<number>(10 * 1024 * 1024); // Default to 10MB until config loads
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatSidebarRef = useRef<ChatSidebarRef>(null);
  const pdfViewerRef = useRef<PDFViewerRef>(null);

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

  const handleDeleteConversation = useCallback((deletedConversationId: string, documentId: string) => {
    if (deletedConversationId === conversationId) {
      // Only clear state if the deleted conversation is the current one
      setCurrentPDF('');
      setDocumentId('');
      setConversationId(null);
      setMessages([]);
      setCurrentAnnotations([]);
      pdfViewerRef.current?.clearAnnotations();
      // update url to remove the chat parameter
      const params = new URLSearchParams(searchParams.toString());
      params.delete('chat');
      router.push(`${pathname}${params.toString() ? '?' + params.toString() : ''}`, {scroll: false});
    }
  }, [conversationId, pathname, router, searchParams])

  const handleCreateNewConversation = useCallback(async (documentId: string) => {
    try {
      // Create new conversation
      const response = await conversationApi.create(documentId);
      if (!response.ok) {
        throw new Error('Failed to create conversation');
      }

      const newConversation = await response.json();

      // Get document info to set PDF URL
      const docResponse = await documentApi.get(documentId);
      if (!docResponse.ok) {
        throw new Error('Failed to fetch document');
      }
      const docData = await docResponse.json();

      // Set the new conversation as active
      setConversationId(newConversation.id);
      setDocumentId(documentId);
      setCurrentPDF(docData.url);
      setMessages([]);
      setCurrentAnnotations([]);
      pdfViewerRef.current?.clearAnnotations();

      // Update URL
      updateUrl(newConversation.id);
    } catch (error) {
      console.error('Error creating new conversation: ', error);
      throw error; // Re-throw so ChatSidebar can handle it
    }
  }, [updateUrl])


  const handleSelectConversation = useCallback(async (convoId: string, docId: string) => {
    if(convoId === conversationId) return;  // Already selected

    try {
      const response = await conversationApi.get(convoId);
      if (!response.ok) {
        throw new Error("Failed to fetch conversation");
      }

      const data = await response.json();
      console.log('[Dashboard] Loaded conversation:', data);

      // update state with the selected conversation
      setConversationId(convoId);
      setDocumentId(docId);
      setCurrentPDF(data.conversation.document.url);
      setMessages(data.messages);

      // Check if the last assistant message has annotations
      const lastAssistantMessage = [...data.messages].reverse().find(
        (m: ChatMessage) => m.role === 'assistant' && m.annotations && m.annotations.length > 0
      );

      if (lastAssistantMessage?.annotations) {
        console.log('[Dashboard] Found annotations in loaded conversation:', lastAssistantMessage.annotations);
        setCurrentAnnotations(lastAssistantMessage.annotations);
      } else {
        // Clear previous annotations when switching conversations
        setCurrentAnnotations([]);
        pdfViewerRef.current?.clearAnnotations();
      }

      // Update URL with the selected conversation
      updateUrl(convoId);
    } catch (error) {
      // TODO: Display error message to user
      console.error('Error loading conversation: ', error);
      setError('Failed to load conversation');
    }
  }, [conversationId, updateUrl]);

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

  // fetch user ID and email on component mount
  useEffect(() => {
    const fetchUser = async () => {
      try {
        const response = await authApi.getUser();
        const data = await response.json();
        if(response.ok) {
          setUserId(data.id);
          setUserEmail(data.email || '');
        } else {
          // TODO: Display error message to user
          console.error('Failed to fetch user: ', data);
        }
      } catch (error) {
        // TODO: Display error message to user
        console.error('Error fetching user:', error);
      }
    };
    fetchUser();
  }, []);

  // fetch config (e.g., max file size) on component mount
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await configApi.get();
        if (response.ok) {
          const data = await response.json();
          setMaxFileSize(data.max_file_size);
        } else {
          console.warn('Failed to fetch config, using default max file size');
        }
      } catch (error) {
        console.warn('Error fetching config, using default max file size:', error);
      }
    };
    fetchConfig();
  }, []);

  // Now effect to restore chat from URL parameter
  useEffect(() => {
    const chatId = searchParams.get('chat');

    // Reset state if no chat ID is present in the URL
    if (!chatId) {
      setCurrentPDF('');
      setDocumentId('');
      setConversationId(null);
      setMessages([]);
      return;
    }

      // Only attempt to restore if we have a chat ID and we're not already showing that conversation
      // and we're not currently loading
      if (chatId && chatId !== conversationId && !isLoading) {
        // find the document ID for this conversation
        const restoreConversation = async () => {
          setIsLoading(true);
          try {
            // first get all conversations to find the document ID matching this chat ID
            const data = await fetchConversations();
            const matchingConversation = data.find(
              (convo: {id: string}) => convo.id === chatId
            );

            if(matchingConversation) {
              // Now we can load this specific conversation
              await handleSelectConversation(chatId, matchingConversation.document_id);
            }
          } catch (error) {
            // TODO: Display error message to user
            console.error('Error restoring conversations: ', error);
            setError('Failed to restore conversation from URL');
          } finally {
            setIsLoading(false);
          }
        };
        restoreConversation();
      }
  }, [searchParams]); // Only depend on searchParams changes

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

    // Validate file size using config from backend
    if (file.size > maxFileSize) {
      const maxSizeMB = (maxFileSize / (1024 * 1024)).toFixed(0);
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
      setCurrentPDF(data.url);
      setDocumentId(data.id);

      // reset messages and annotations for new document
      setMessages([]);
      setCurrentAnnotations([]);
      pdfViewerRef.current?.clearAnnotations();

      // set new conversation id from the response
      if(data.conversationId){
        setConversationId(data.conversationId);
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

  const handleSendMessage = async (content: string, model: string) => {
    if (!content.trim() || !conversationId || !documentId) return;

    const tempUserMessageId = `temp-user-${Date.now()}`;
    const tempAssistantMessageId = `temp-assistant-${Date.now()}`;

    const userMessage = {
      id: tempUserMessageId,
      role: 'user' as const,
      content: content.trim()
    };

    // Add user message to chat
    setMessages(prev => [...prev, userMessage]);

    // Create temporary assistant message for streaming
    const assistantMessage: ChatMessage = {
      id: tempAssistantMessageId,
      role: 'assistant',
      content: '',
      annotations: undefined
    };
    setMessages(prev => [...prev, assistantMessage]);
    setIsLoading(true);
    setError(null);

    try {
      await chatApi.sendMessageStream(
        conversationId,
        documentId,
        content,
        model,
        // onChunk - update assistant message content incrementally
        (chunk: string) => {
          setMessages(prev => prev.map(msg =>
            msg.id === tempAssistantMessageId
              ? { ...msg, content: msg.content + chunk }
              : msg
          ));
        },
        // onDone - replace temp messages with final messages
        (data: any) => {
          console.log('[Dashboard] Stream completed:', data);

          setMessages(prev => [
            ...prev.filter(m => m.id !== tempUserMessageId && m.id !== tempAssistantMessageId),
            {
              id: data.user_message.id,
              role: data.user_message.role as 'user',
              content: data.user_message.content,
              annotations: undefined
            },
            {
              id: data.assistant_message.id,
              role: data.assistant_message.role as 'assistant',
              content: data.assistant_message.content,
              annotations: data.assistant_message.annotations
            }
          ]);

          // If the assistant message has annotations, update the PDF viewer
          if (data.assistant_message?.annotations && data.assistant_message.annotations.length > 0) {
            console.log('[Dashboard] Processing annotations:', data.assistant_message.annotations);
            setCurrentAnnotations(data.assistant_message.annotations);

            // Auto-navigate to the first annotation's page
            const firstAnnotation = data.assistant_message.annotations[0];
            if (pdfViewerRef.current && firstAnnotation) {
              console.log('[Dashboard] Navigating to page:', firstAnnotation.pageNumber);
              pdfViewerRef.current.goToPage(firstAnnotation.pageNumber);
              if (firstAnnotation.sourceText) {
                console.log('[Dashboard] Highlighting text:', firstAnnotation.sourceText);
                // Small delay to allow page to render
                setTimeout(() => {
                  pdfViewerRef.current?.highlightText(firstAnnotation.pageNumber, firstAnnotation.sourceText);
                }, 500);
              }
            }
          } else {
            console.log('[Dashboard] No annotations in response');
          }

          // Refresh sidebar to update conversation title (if this was the first message)
          const messageCountBefore = messages.length;
          if (messageCountBefore === 0) {
            // This was the first message, title should have been generated
            // Refresh sidebar after a short delay to allow backend to save the title
            setTimeout(() => {
              if (chatSidebarRef.current) {
                const sidebar = chatSidebarRef.current as unknown as { refreshConversations: () => void };
                if (typeof sidebar.refreshConversations === 'function') {
                  sidebar.refreshConversations();
                }
              }
            }, 1000);
          }

          setIsLoading(false);
          setError(null);
        },
        // onError - handle errors
        (errorMessage: string) => {
          console.error('Streaming error:', errorMessage);
          // Remove temporary messages on error
          setMessages(prev => prev.filter(m =>
            m.id !== tempUserMessageId && m.id !== tempAssistantMessageId
          ));
          setIsLoading(false);
          setError(errorMessage);
        }
      );
    } catch (error) {
      // Remove the temporary messages on error
      setMessages(prev => prev.filter(m =>
        m.id !== tempUserMessageId && m.id !== tempAssistantMessageId
      ));
      setIsLoading(false);
      const errorMessage = error instanceof Error ? error.message : 'Failed to send message';
      console.error('Chat error: ', errorMessage);
      setError(errorMessage);
    }
  }

  // Handle annotation click from chat - navigate to page and highlight text
  const handleAnnotationClick = useCallback((annotation: AnnotationReference) => {
    if (pdfViewerRef.current) {
      // Navigate to the page
      pdfViewerRef.current.goToPage(annotation.pageNumber);

      // Set the annotation to be displayed
      setCurrentAnnotations([annotation]);

      // If there's text to highlight, use the highlight function
      if (annotation.sourceText) {
        pdfViewerRef.current.highlightText(annotation.pageNumber, annotation.sourceText);
      }
    }
  }, []);

  // Resizer handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing || !containerRef.current) return;

      const containerRect = containerRef.current.getBoundingClientRect();
      const newPosition = ((e.clientX - containerRect.left) / containerRect.width) * 100;

      // Constrain between 20% and 80% to prevent components from becoming too small
      const constrainedPosition = Math.max(20, Math.min(80, newPosition));
      setSplitPosition(constrainedPosition);
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
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
        userId={userId}
        userEmail={userEmail}
        onSelectConversation={handleSelectConversation}
        currentConversationId={conversationId}
        onDeleteConversation={handleDeleteConversation}
        onCreateNewConversation={handleCreateNewConversation}
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
            currentPDF={currentPDF}
            onFileUpload={handleFileUpload}
            fileInputRef={fileInputRef as React.RefObject<HTMLInputElement>}
            annotations={currentAnnotations}
            onAnnotationClick={handleAnnotationClick}
          />
        </div>

        {/* Resizer */}
        <div
          onMouseDown={handleMouseDown}
          className={`absolute top-0 bottom-0 w-1 bg-gray-200 hover:bg-blue-500 cursor-col-resize transition-colors z-20 ${
            isResizing ? 'bg-blue-500' : ''
          }`}
          style={{ left: `${splitPosition}%`, transform: 'translateX(-50%)' }}
        >
          <div className={`absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-8 h-12 rounded-full flex items-center justify-center transition-all ${
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
            messages={messages}
            onSendMessage={handleSendMessage}
            onVoiceRecord={handleVoiceRecord}
            isConversationSelected={!!conversationId}
            onAnnotationClick={handleAnnotationClick}
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
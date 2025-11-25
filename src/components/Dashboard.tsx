// app/components/Dashboard.tsx
"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react"
import EnhancedPDFViewer from "./EnhancedPDFViewer";
import ChatInterface from "./ChatInterface";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import ChatSidebar, { ChatSidebarRef } from "./ChatSidebar";
import { authApi, documentApi, chatApi, conversationApi } from "@/lib/api-client";

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
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
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatSidebarRef = useRef<ChatSidebarRef>(null);

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

  const handleDeleteConversation = useCallback((deletedConversationId: string) => {
    if (deletedConversationId === conversationId) {
      setCurrentPDF('');
      setDocumentId('');
      setConversationId(null);
      setMessages([]);
      // update url to remove the chat parameter
      const params = new URLSearchParams(searchParams.toString());
      params.delete('chat');
      router.push(`${pathname}${params.toString() ? '?' + params.toString() : ''}`, {scroll: false});
    }
  }, [conversationId, pathname, router, searchParams])


  const handleSelectConversation = useCallback(async (convoId: string, docId: string) => {
    if(convoId === conversationId) return;  // Already selected

    try {
      const response = await conversationApi.get(convoId);
      if (!response.ok) {
        throw new Error("Failed to fetch conversation");
      }

      const data = await response.json();

      // update state with the selected conversation
      setConversationId(convoId);
      setDocumentId(docId);
      setCurrentPDF(data.conversation.document.url);
      setMessages(data.messages);

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

    //  validate file size(e.g., 10MB limit)
    const MAX_SIZE = 10 * 1024 * 1024;
    if (file.size > MAX_SIZE) {
      setError('File size should be less than 10MB');
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

      // reset messages for new document
      setMessages([]);

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

    const userMessage = {
      id: `temp=${Date.now()}`,
      role: 'user' as const,
      content: content.trim()
    };

    // add user message to chat
    setMessages(prev => [...prev, userMessage]);

    try {
      const response = await chatApi.sendMessage(conversationId, documentId, content, model);
      console.log(response)
      if(!response.ok) {
        // Try to extract error message from response
        let errorMessage = 'Failed to send message';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.error || errorMessage;
        } catch {
          // If response is not JSON, use status text
          errorMessage = response.statusText || errorMessage;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();

      // update messages with server response
      setMessages(prev => [
        ...prev.filter(m=>m.id !== userMessage.id), // Remove temp message
        data.user_message,  // Add actual user message with ID
        data.assistant_message // Add assistant response
      ]);

      // Clear any previous errors on success
      setError(null);
    } catch (error) {
      // Remove the temporary user message on error
      setMessages(prev => prev.filter(m => m.id !== userMessage.id));

      const errorMessage = error instanceof Error ? error.message : 'Failed to send message';
      console.error('Chat error: ', errorMessage);
      setError(errorMessage);
    }
  }

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
      />

      {/* Main content Area */}
      <div ref={containerRef} className="flex flex-1 overflow-hidden relative">
        {/* PDF Viewer Section */}
        <div
          className="h-full overflow-hidden"
          style={{ width: `${splitPosition}%` }}
        >
          <EnhancedPDFViewer
            currentPDF={currentPDF}
            onFileUpload={handleFileUpload}
            fileInputRef={fileInputRef as React.RefObject<HTMLInputElement>}
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
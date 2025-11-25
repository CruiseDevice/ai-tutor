// app/components/Dashboard.tsx
"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react"
import EnhancedPDFViewer from "./EnhancedPDFViewer";
import ChatInterface from "./ChatInterface";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import ChatSidebar, { ChatSidebarRef } from "./ChatSidebar";
import { authApi, documentApi, chatApi, conversationApi, getJson } from "@/lib/api-client";

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
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatSidebarRef = useRef<ChatSidebarRef>(null);

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
  
  // fetch user ID on component mount
  useEffect(() => {
    const fetchUserId = async () => {
      try {
        const response = await authApi.getUser();
        const data = await response.json();
        if(response.ok) {
          setUserId(data.id);
        } else {
          // TODO: Display error message to user
          console.error('Failed to fetch user: ', data);
        }
      } catch (error) {
        // TODO: Display error message to user
        console.error('Error fetching user:', error);
      }
    };
    fetchUserId();
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

      if(!response.ok) {
        throw new Error('Failed to send message');
      }

      const data = await response.json();
      
      // update messages with server response
      setMessages(prev => [
        ...prev.filter(m=>m.id !== userMessage.id), // Remove temp message
        data.userMessage,  // Add actual user message with ID
        data.assistantMessage // Add assistant response
      ]);
    } catch (error) {
      // TODO: Display error message to user
      console.error('Chat error: ', error);
      setError('Failed to send message');
    }
  }

  if (error) {
    return <div className="p-4 text-red-500">{error}</div>
  }

  return (
    <div className="h-screen flex overflow-hidden">
      {/* Sidebar */}
      <ChatSidebar
        ref={chatSidebarRef}
        userId={userId}
        onSelectConversation={handleSelectConversation}
        currentConversationId={conversationId}
        onDeleteConversation={handleDeleteConversation}
      />

      {/* Main content Area */}
      <div className="flex flex-1 overflow-hidden">
        {/* PDF Viewer Section */}
        <div className="w-3/5">
          <EnhancedPDFViewer
            currentPDF={currentPDF}
            onFileUpload={handleFileUpload}
            fileInputRef={fileInputRef as React.RefObject<HTMLInputElement>}
          />
        </div>
        {/* Chat Section */}
        <ChatInterface
          messages={messages}
          onSendMessage={handleSendMessage}
          onVoiceRecord={handleVoiceRecord}
          isConversationSelected={!!conversationId}
        />
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
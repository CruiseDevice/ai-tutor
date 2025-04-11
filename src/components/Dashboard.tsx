// app/components/Dashboard.tsx
"use client";

import { Suspense, useCallback, useEffect, useState } from "react"
import EnhancedPDFViewer from "./EnhancedPDFViewer";
import ChatInterface from "./ChatInterface";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import ChatSidebar from "./ChatSidebar";

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
      const response = await fetch(`/api/conversations/${convoId}`);
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
      console.error('Error loading conversations: ', error);
      setError('Failed to load conversation');
    }
  }, [conversationId, updateUrl]);

  useEffect(() => {
    // debugging log
    const checkAuth = async () => {
      try {
        const response = await fetch('/api/auth/verify-session');

        if(!response.ok) {
          router.push('/login')
        }
      } catch (error) {
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
        const response = await fetch('/api/auth/user');
        const data = await response.json();
        if(response.ok) {
          setUserId(data.id);
        } else {
          console.error('Failed to fetch user: ', data);
        }
      } catch (error) {
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
    if (chatId && chatId !== conversationId && !isLoading) {
      // find the document ID for this conversation
      const restoreConversation = async () => {
        setIsLoading(true);
        try {
          // first get all conversations to find the document ID matching this chat ID
          const response = await fetch('/api/conversations');
          if(!response.ok) {
            throw new Error('Failed to fetch conversations');
          }
          const data = await response.json();
          const matchingConversation = data.conversations.find(
            (convo: {id: string}) => convo.id === chatId
          );

          if(matchingConversation) {
            // Now we can load this specific conversation
            await handleSelectConversation(chatId, matchingConversation.documentId);
          }
        } catch (error) {
          console.error('Error restoring conversations: ', error);
          setError('Failed to restore conversation from URL');
        } finally {
          setIsLoading(false);
        }
      };
      restoreConversation();
    }
  }, [searchParams, userId, conversationId, handleSelectConversation, isLoading]);

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

    // create form data
    const formData = new FormData();
    formData.append('file', file);

    try{
      // upload file
      const response = await fetch('/api/documents', {
        method: 'POST',
        body: formData,
      });

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
      } else {
        console.error('No conversationId returned from server');
      }
    } catch (error) {
      console.error('Upload error:', error);
      setError('Failed to upload document');
    }
  }

  const handleVoiceRecord = () => {
    // implement voice recording logic later
    console.log('Voice recording toggled');
  }

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || !conversationId) return;

    const userMessage = {
      id: `temp=${Date.now()}`,
      role: 'user' as const,
      content: content.trim()
    };

    // add user message to chat 
    setMessages(prev => [...prev, userMessage]);

    try {
      const payload = {
        content,
        conversationId,
        documentId
      };

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

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
        userId={userId}
        onSelectConversation={handleSelectConversation}
        currentConversationId={conversationId}
        onDeleteConversation={handleDeleteConversation}
      />

      {/* Main content Area */}
      <div className="flex flex-1 overflow-hidden">
        {/* PDF Viewer Section */}
        <EnhancedPDFViewer 
          currentPDF={currentPDF}
          onFileUpload={handleFileUpload}
        />
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
// src/components/ChatSidebar.tsx

import { ChevronLeft, ChevronRight, FileText, MessageSquare, Plus, Search, Trash2 } from "lucide-react";
import { forwardRef, useEffect, useImperativeHandle, useState } from "react";

interface Conversation {
  id: string;
  documentId: string;
  title: string;  // document title
  updatedAt: string
}

interface ChatSidebarProps {
  userId: string;
  onSelectConversation: (conversationId: string, documentId: string) => void;
  currentConversationId: string | null;
  onDeleteConversation?: (conversationId: string, documentId: string) => void;
  onUploadDocument?: () => void; // New prop for handling document upload
}

// Create an interface for the ref
export interface ChatSidebarRef {
  addNewConversation: (newConversation: Conversation) => void;
}

const ChatSidebar = forwardRef<ChatSidebarRef, ChatSidebarProps>(({
  userId,
  onSelectConversation,
  currentConversationId,
  onDeleteConversation,
  onUploadDocument,
}, ref) => {
  const [isOpen, setIsOpen] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filteredConversations, setFilteredConversations] = useState<Conversation[]>([]);
  
  // Method to add a new conversation to the list
  function addNewConversation(newConversation: Conversation) {
    setConversations(prev => [newConversation, ...prev]);
    // If no search query is active, also update the filtered list
    if (!searchQuery) {
      setFilteredConversations(prev => [newConversation, ...prev]);
    } else if (newConversation.title.toLowerCase().includes(searchQuery.toLowerCase())) {
      // If there's a search and the new conversation matches it, add it to filtered results
      setFilteredConversations(prev => [newConversation, ...prev]);
    }
  }
  
  // Expose the method to parent components using useImperativeHandle
  useImperativeHandle(ref, () => ({
    addNewConversation
  }));

  useEffect(() => {
    if (!userId) return;

    const fetchConversations = async () => {
      setIsLoading(true);
      try {
        const response = await fetch('/api/conversations');
        if (!response.ok) {
          throw new Error('Failed to fetch conversations');
        }
        
        const data = await response.json();
        setConversations(data.conversations);
        setFilteredConversations(data.conversations);
      } catch (error) {
        console.error('Error fetching conversations: ', error)
      } finally {
        setIsLoading(false);
      }
    };

    fetchConversations();
  }, [userId]);

  // Filter conversations based on search query
  useEffect(() => {
    if (!searchQuery.trim()) {
      setFilteredConversations(conversations);
      return;
    }
    
    const filtered = conversations.filter(conversation => 
      conversation.title.toLowerCase().includes(searchQuery.toLowerCase())
    );
    setFilteredConversations(filtered);
  }, [searchQuery, conversations]);

  const handleDelete = async(e: React.MouseEvent, conversationId: string, documentId: string) => {
    e.stopPropagation();  // Prevent triggering the selection
    if (confirm("Are you sure you want to delete this conversation? This will also delete the associated document.")) {
      setDeleting(conversationId);

      try {
        const response = await fetch(`/api/conversations/${conversationId}`, {
          method: 'DELETE',
        });

        if (!response.ok) {
          throw new Error('Failed to delete conversation');
        }

        // Remove from local state
        setConversations(conversations.filter(c => c.id != conversationId));
        setFilteredConversations(filteredConversations.filter(c => c.id != conversationId));

        // If the current conversation is deleted, call the parent handler
        if (currentConversationId === conversationId && onDeleteConversation) {
          onDeleteConversation(conversationId, documentId);
        }
      } catch (error) {
        console.error('Error deleting conversation:', error);
        alert('Failed to delete conversation');
      } finally {
        setDeleting(null);
      }
    }
  }

  // Format date for display
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    
    // Check if it's today
    if (date.toDateString() === now.toDateString()) {
      return `Today at ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    }
    
    // Check if it's yesterday
    const yesterday = new Date(now);
    yesterday.setDate(now.getDate() - 1);
    if (date.toDateString() === yesterday.toDateString()) {
      return `Yesterday at ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    }
    
    // If it's in the last week, show the day name
    if (now.getTime() - date.getTime() < 7 * 24 * 60 * 60 * 1000) {
      return date.toLocaleDateString(undefined, { weekday: 'long' });
    }
    
    // Otherwise show full date
    return date.toLocaleDateString();
  };

  return (
    <div 
      className={`h-full bg-gray-100 border-r border-gray-200 transition-all duration-300 flex flex-col ${
        isOpen ? 'w-72' : 'w-16'
      }`}
    >
      {/* Header with toggle button */}
      <div className="flex items-center justify-between p-3 border-b border-gray-200 bg-gray-200">
        {isOpen && <h2 className="font-semibold text-lg text-gray-700">Documents</h2>}
        <button 
          onClick={() => setIsOpen(!isOpen)}
          className="p-2 rounded-full hover:bg-gray-300 text-gray-500 hover:text-gray-700 transition-colors"
          aria-label={isOpen ? "Collapse sidebar" : "Expand sidebar"}
        >
          {isOpen ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
        </button>
      </div>

      {/* Sidebar content - only shown when open */}
      {isOpen && (
        <>
          {/* Search and Upload controls */}
          <div className="p-3 border-b border-gray-200">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search size={16} className="text-gray-400" />
              </div>
              <input
                type="text"
                placeholder="Search documents..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
              />
            </div>
            <button
              onClick={onUploadDocument}
              className="mt-3 w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg transition-colors text-sm font-medium"
            >
              <Plus size={16} />
              Upload New Document
            </button>
          </div>

          {/* Conversation list */}
          <div className="flex-1 overflow-y-auto">
            {isLoading ? (
              <div className="flex justify-center items-center h-32">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
              </div>
            ) : filteredConversations.length === 0 ? (
              <div className="text-center p-6 text-gray-500">
                {searchQuery ? "No matching documents found" : "No documents yet"}
              </div>
            ) : (
              <ul className="p-3 space-y-1">
                {filteredConversations.map((conversation) => (
                  <li key={conversation.id} className="group">
                    <div className={`relative rounded-lg transition-colors ${
                      currentConversationId === conversation.id 
                        ? 'bg-blue-100 hover:bg-blue-200' 
                        : 'hover:bg-gray-200'
                    }`}>
                      <button
                        onClick={() => onSelectConversation(conversation.id, conversation.documentId)}
                        className="w-full text-left p-3 pr-9 rounded-lg flex items-start gap-3"
                      >
                        <FileText size={18} className={`flex-shrink-0 mt-0.5 ${
                          currentConversationId === conversation.id ? 'text-blue-600' : 'text-gray-500'
                        }`}/>
                        <div className="overflow-hidden flex-1">
                          <div className="truncate font-medium text-sm">
                            {conversation.title}
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            {formatDate(conversation.updatedAt)}
                          </div>
                        </div>
                      </button>
                      
                      <button
                        onClick={(e) => handleDelete(e, conversation.id, conversation.documentId)}
                        className={`absolute right-2 top-3 p-1.5 rounded-full ${
                          deleting === conversation.id 
                            ? 'bg-red-100' 
                            : 'opacity-0 group-hover:opacity-100 hover:bg-red-100'
                        } transition-opacity`}
                        disabled={deleting === conversation.id}
                        title="Delete document and conversation"
                      >
                        {deleting === conversation.id ? (
                          <div className="h-4 w-4 border-2 border-t-red-500 rounded-full animate-spin"></div>
                        ) : (
                          <Trash2 size={16} className="text-red-500"/>
                        )}
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Footer with user info */}
          <div className="p-3 border-t border-gray-200 bg-gray-200">
            <div className="flex items-center text-sm text-gray-600">
              <MessageSquare size={16} className="mr-2" />
              <span className="truncate">{userId ? `User: ${userId.substring(0, 8)}...` : 'Not logged in'}</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
});

// Add display name for better debugging
ChatSidebar.displayName = 'ChatSidebar';

export default ChatSidebar;
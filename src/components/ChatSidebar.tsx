// src/components/ChatSidebar.tsx
import { ChevronDown, ChevronLeft, ChevronRight, FileText, LogOut, Settings, Trash2, User } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

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
}

// TODO: Not sure if this is needed
// Create an interface for the ref
export interface ChatSidebarRef {
  addNewConversation: (newConversation: Conversation) => void;
}

const ChatSidebar = forwardRef<ChatSidebarRef, ChatSidebarProps>(({
  userId,
  onSelectConversation,
  currentConversationId,
  onDeleteConversation,
}, ref) => {
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  
  // Method to add a new conversation to the list
  function addNewConversation(newConversation: Conversation) {
    setConversations(prev => [newConversation, ...prev]);
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
      } catch (error) {
        console.error('Error fetching conversations: ', error)
      } finally {
        setIsLoading(false);
      }
    };

    fetchConversations();
  }, [userId]);

  // handle clicks outside the dropdown to close it
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsMenuOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleLogout = async () => {
    try {
      const response = await fetch('/api/auth/logout', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to logout');
      }

      // Redirect to login page after successful logout 
      router.push('/login')
      // Force a page refresh to clear any client-side state
      router.refresh();
    } catch (error) {
      //TODO: Display error message to user
      console.error('Logout error: ', error);
    }
  };

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

        // If the current conversation is deleted, call the parent handler
        if (currentConversationId === conversationId && onDeleteConversation) {
          onDeleteConversation(conversationId, documentId);
        }
      } catch (error) {
        //TODO: Display error message to user
        console.error('Delete conversation error: ', error);
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
          {/* Conversation list */}
          <div className="flex-1 overflow-y-auto">
            {isLoading ? (
              <div className="flex justify-center items-center h-32">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
              </div>
            ) : conversations.length === 0 ? (
              <div className="text-center p-6 text-gray-500">
                No documents yet
              </div>
            ) : (
              <ul className="p-3 space-y-1">
                {conversations.map((conversation) => (
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

         {/* Footer with user info and dropdown menu */}
         <div className="relative p-3 border-t border-gray-200 bg-gray-200">
          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className="w-full flex items-center justify-between text-sm text-gray-600 hover:bg-gray-300 p-2 rounded"
          >
            <div className="flex items-center">
              <User size={16} className="mr-2"/>
              <span className="truncate">{userId ? `User: ${userId.substring(0, 8)}...` : 'Not logged in'}</span>
            </div>
            <ChevronDown size={16} className={`transition-transform duration-200 ${isMenuOpen ? 'rotate-180' : ''}`}/>
          </button>
          {/* Dropdown menu */}
          {isMenuOpen && (
            <div 
              ref={menuRef}
              className="absolute bottom-14 left-3 right-3 bg-white rounded-md shadow-lg border border-gray-200 overflow-hidden z-10"
            >
              <Link 
                href="/settings"
                className="flex items-center p-3 text-sm text-gray-700 hover:bg-gray-100 transition-colors"
              >
                <Settings size={16} className="mr-2 text-gray-500" />
                API Settings
              </Link>
              <button 
                onClick={handleLogout}
                className="w-full flex items-center p-3 text-sm text-red-600 hover:bg-gray-100 transition-colors"
              >
                <LogOut size={16} className="mr-2" />
                Logout
              </button>
            </div>
          )}
         </div>
        </>
      )}
    </div>
  );
});

// Add display name for better debugging
ChatSidebar.displayName = 'ChatSidebar';

export default ChatSidebar;
// src/components/ChatSidebar.tsx
import { ChevronDown, ChevronLeft, ChevronRight, FileText, LogOut, Plus, Settings, Trash2, MessageSquare } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState, useCallback } from "react";
import { conversationApi, authApi } from "@/lib/api-client";

interface Conversation {
  id: string;
  documentId: string;
  title: string;
  updatedAt: string
}

interface BackendConversationResponse {
  id: string;
  user_id: string;
  document_id: string;
  title: string | null;  // Smart conversation title
  created_at: string;
  updated_at: string;
  document: {
    id: string;
    title: string;
    url: string;
  } | null;
}

interface ChatSidebarProps {
  userId: string;
  userEmail?: string;
  onSelectConversation: (conversationId: string, documentId: string) => void;
  currentConversationId: string | null;
  onDeleteConversation?: (conversationId: string, documentId: string) => void;
}

export interface ChatSidebarRef {
  addNewConversation: (newConversation: Conversation) => void;
  refreshConversations: () => void;
}

const ChatSidebar = forwardRef<ChatSidebarRef, ChatSidebarProps>(({
  userId,
  userEmail,
  onSelectConversation,
  currentConversationId,
  onDeleteConversation,
}, ref) => {
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(true); // Default to open for better UX
  const [isLoading, setIsLoading] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  function addNewConversation(newConversation: Conversation) {
    setConversations(prev => [newConversation, ...prev]);
  }

  const refreshConversations = useCallback(async () => {
    if (!userId) return;

    setIsLoading(true);
    try {
      const response = await conversationApi.list();
      if (!response.ok) {
        throw new Error('Failed to fetch conversations');
      }

      const data = await response.json() as BackendConversationResponse[];
      const mappedConversations = data.map((conv) => ({
        id: conv.id,
        documentId: conv.document_id,
        // Use smart conversation title, fallback to document title, then default
        title: conv.title || conv.document?.title || 'Untitled Document',
        updatedAt: conv.updated_at
      }));
      setConversations(mappedConversations);
    } catch (error) {
      console.error('Error refreshing conversations: ', error)
    } finally {
      setIsLoading(false);
    }
  }, [userId]);

  useImperativeHandle(ref, () => ({
    addNewConversation,
    refreshConversations
  }));

  useEffect(() => {
    refreshConversations();
  }, [refreshConversations]);

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
        const response = await authApi.logout();

      if (!response.ok) {
        throw new Error('Failed to logout');
      }

      router.push('/login')
      router.refresh();
    } catch (error) {
      console.error('Logout error: ', error);
    }
  };

  const handleDelete = async(e: React.MouseEvent, conversationId: string, documentId: string) => {
    e.stopPropagation();
    if (confirm("Are you sure you want to delete this conversation? This will also delete the associated document.")) {
      setDeleting(conversationId);

      try {
        const response = await conversationApi.delete(conversationId);

        if (!response.ok) {
          throw new Error('Failed to delete conversation');
        }

        setConversations(conversations.filter(c => c.id != conversationId));

        if (currentConversationId === conversationId && onDeleteConversation) {
          onDeleteConversation(conversationId, documentId);
        }
      } catch (error) {
        console.error('Delete conversation error: ', error);
        alert('Failed to delete conversation');
      } finally {
        setDeleting(null);
      }
    }
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();

    if (date.toDateString() === now.toDateString()) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    if (now.getTime() - date.getTime() < 7 * 24 * 60 * 60 * 1000) {
      return date.toLocaleDateString(undefined, { weekday: 'short' });
    }

    return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  };

  const handleNavigateToDashboard = (e: React.MouseEvent) => {
    e.preventDefault();
    router.push('/dashboard');
  }

  return (
    <div
      className={`h-full bg-[#0f172a] text-slate-300 border-r border-slate-800 transition-all duration-300 flex flex-col relative z-20 ${
        isOpen ? 'w-80' : 'w-[4.5rem]'
      }`}
    >
      {/* Header */}
      <div className={`flex items-center p-4 mb-2 ${isOpen ? 'justify-between' : 'justify-center'}`}>
        {isOpen && (
          <div className="flex items-center gap-3 overflow-hidden">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-blue-600 to-violet-600 flex items-center justify-center shadow-lg shadow-blue-900/20 flex-shrink-0">
              <FileText className="text-white w-4 h-4" />
            </div>
            <h2 className="font-bold text-lg text-white tracking-tight whitespace-nowrap">
              Tutor
              <span className="text-blue-500">.ai</span>
            </h2>
          </div>
        )}
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="p-2 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-all duration-200 active:scale-95"
          aria-label={isOpen ? "Collapse sidebar" : "Expand sidebar"}
        >
          {isOpen ? <ChevronLeft size={18} /> : <ChevronRight size={18} />}
        </button>
      </div>

      {/* New Chat Action */}
      <div className={`px-3 mb-6 ${isOpen ? '' : 'flex justify-center'}`}>
        <button
          onClick={handleNavigateToDashboard}
          className={`group flex items-center gap-3 bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 text-white rounded-xl shadow-lg shadow-blue-900/20 transition-all duration-300 hover:shadow-blue-900/40 hover:-translate-y-0.5 ${
            isOpen ? 'w-full py-3 px-4' : 'w-10 h-10 justify-center p-0'
          }`}
        >
          <Plus size={20} className={`${isOpen ? '' : 'ml-0'} transition-transform group-hover:rotate-90`} />
          {isOpen && <span className="font-semibold text-sm">New Study Session</span>}
        </button>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto px-3 pb-2 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-slate-700 [&::-webkit-scrollbar-thumb]:rounded-full hover:[&::-webkit-scrollbar-thumb]:bg-slate-600">
        {isOpen && (
          <div className="px-2 mb-2 text-xs font-semibold text-slate-500 uppercase tracking-wider">
            Recent Documents
          </div>
        )}

        {isLoading ? (
          <div className="flex justify-center items-center h-20">
            <div className="animate-spin rounded-full h-6 w-6 border-2 border-slate-600 border-t-blue-500"></div>
          </div>
        ) : conversations.length === 0 ? (
          <div className={`text-center p-4 text-slate-500 text-sm ${!isOpen && 'hidden'}`}>
            <div className="mb-2 inline-flex p-3 rounded-full bg-slate-800/50">
              <MessageSquare size={20} />
            </div>
            <p>No documents yet</p>
          </div>
        ) : (
          <ul className="space-y-1">
            {conversations.map((conversation) => (
              <li key={conversation.id} className="group relative">
                <button
                  onClick={() => onSelectConversation(conversation.id, conversation.documentId)}
                  className={`w-full text-left rounded-xl transition-all duration-200 flex items-center group/item ${
                    currentConversationId === conversation.id
                      ? 'bg-slate-800 text-white shadow-sm ring-1 ring-slate-700/50'
                      : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                  } ${isOpen ? 'p-3' : 'p-3 justify-center'}`}
                >
                  <div className={`relative flex-shrink-0 ${isOpen ? 'mr-3' : ''}`}>
                    <FileText size={18} className={currentConversationId === conversation.id ? 'text-blue-400' : 'text-slate-500 group-hover/item:text-slate-400'} />
                    {currentConversationId === conversation.id && (
                      <span className="absolute -right-1 -top-1 w-2 h-2 rounded-full bg-blue-500 ring-2 ring-[#0f172a]" />
                    )}
                  </div>

                  {isOpen && (
                    <div className="overflow-hidden flex-1">
                      <div className="truncate font-medium text-sm leading-tight mb-0.5">
                        {conversation.title}
                      </div>
                      <div className="flex items-center justify-between text-[10px] text-slate-500">
                        <span>{formatDate(conversation.updatedAt)}</span>
                      </div>
                    </div>
                  )}
                </button>

                {/* Hover Actions (Delete) */}
                {isOpen && (
                  <div className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => handleDelete(e, conversation.id, conversation.documentId)}
                      className={`p-1.5 rounded-lg hover:bg-red-500/10 hover:text-red-400 transition-colors ${
                        deleting === conversation.id ? 'text-red-400 opacity-100' : 'text-slate-500'
                      }`}
                      title="Delete document"
                      disabled={deleting === conversation.id}
                    >
                      {deleting === conversation.id ? (
                        <div className="h-3.5 w-3.5 border-2 border-t-red-500 border-transparent rounded-full animate-spin" />
                      ) : (
                        <Trash2 size={14} />
                      )}
                    </button>
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Footer User Menu */}
      <div className="p-3 border-t border-slate-800 bg-[#0f172a]">
        <div className="relative">
          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className={`w-full flex items-center rounded-xl hover:bg-slate-800 transition-colors ${
              isOpen ? 'p-2 justify-between' : 'p-2 justify-center'
            } ${isMenuOpen ? 'bg-slate-800' : ''}`}
          >
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-emerald-400 to-teal-600 flex items-center justify-center text-white shadow-lg shadow-emerald-900/20 font-semibold text-xs">
                {userId ? userId.substring(0, 2).toUpperCase() : 'U'}
              </div>
              {isOpen && (
                <div className="text-left">
                  <div className="text-xs font-medium text-white truncate w-32">
                    {userEmail || userId || 'User'}
                  </div>
                  <div className="text-[10px] text-slate-500">Pro Plan</div>
                </div>
              )}
            </div>
            {isOpen && (
              <ChevronDown size={14} className={`text-slate-500 transition-transform duration-200 ${isMenuOpen ? 'rotate-180' : ''}`} />
            )}
          </button>

          {/* Popup Menu */}
          {isMenuOpen && (
            <div
              ref={menuRef}
              className={`absolute bottom-full left-0 bg-[#1e293b] rounded-xl shadow-xl shadow-black/50 border border-slate-700 overflow-hidden z-50 mb-2 backdrop-blur-xl ${
                isOpen ? 'w-full min-w-[200px]' : 'left-10 w-48'
              }`}
            >
              <div className="p-1">
                <Link
                  href="/settings"
                  className="flex items-center px-3 py-2 text-sm text-slate-300 hover:bg-slate-700/50 hover:text-white rounded-lg transition-colors gap-2"
                >
                  <Settings size={16} className="text-slate-400" />
                  <span>Settings</span>
                </Link>
                <div className="h-px bg-slate-700/50 my-1" />
                <button
                  onClick={handleLogout}
                  className="w-full flex items-center px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 hover:text-red-300 rounded-lg transition-colors gap-2"
                >
                  <LogOut size={16} />
                  <span>Logout</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

ChatSidebar.displayName = 'ChatSidebar';

export default ChatSidebar;
// src/components/ChatSidebar.tsx
import { ChevronDown, ChevronLeft, ChevronRight, FileText, LogOut, Plus, Settings, Trash2, MessageSquare, ChevronUp } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState, useCallback } from "react";
import { conversationApi, authApi } from "@/lib/api-client";

interface Conversation {
  id: string;
  documentId: string;
  title: string;
  updatedAt: string;
}

interface DocumentGroup {
  document: {
    id: string;
    title: string;
    url: string;
  } | null;
  conversations: Array<{
    id: string;
    user_id: string;
    document_id: string;
    title: string | null;
    created_at: string;
    updated_at: string;
  }>;
}

interface ChatSidebarProps {
  userId: string;
  userEmail?: string;
  onSelectConversation: (conversationId: string, documentId: string) => void;
  currentConversationId: string | null;
  onDeleteConversation?: (conversationId: string, documentId: string) => void;
  onCreateNewConversation?: (documentId: string) => Promise<{ id: string; document_id?: string; title?: string; updated_at?: string } | void>;
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
  onCreateNewConversation,
}, ref) => {
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(true); // Default to open for better UX
  const [isLoading, setIsLoading] = useState(false);
  const [documentGroups, setDocumentGroups] = useState<DocumentGroup[]>([]);
  const [expandedDocuments, setExpandedDocuments] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState<string | null>(null);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [creatingConversation, setCreatingConversation] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  function addNewConversation(newConversation: Conversation) {
    // Find the document group and add the conversation
    setDocumentGroups(prev => {
      const updated = [...prev];
      const groupIndex = updated.findIndex(g => g.document?.id === newConversation.documentId);

      if (groupIndex >= 0) {
        const group = updated[groupIndex];
        // Check if conversation already exists to prevent duplicate keys
        const exists = group.conversations.some(c => c.id === newConversation.id);

        if (!exists) {
          updated[groupIndex] = {
            ...group,
            conversations: [
              {
                id: newConversation.id,
                user_id: userId,
                document_id: newConversation.documentId,
                title: newConversation.title,
                created_at: new Date().toISOString(),
                updated_at: newConversation.updatedAt,
              },
              ...group.conversations
            ]
          };
        }
      }
      return updated;
    });
  }

  const refreshConversations = useCallback(async () => {
    if (!userId) return;

    setIsLoading(true);
    try {
      const response = await conversationApi.list(true); // Use grouped format
      if (!response.ok) {
        throw new Error('Failed to fetch conversations');
      }

      const data = await response.json() as DocumentGroup[];
      console.log('Refreshed conversations:', data);
      setDocumentGroups(data);

      // Expand documents that have the current conversation
      if (currentConversationId) {
        const hasCurrentConversation = data.some(group =>
          group.conversations.some(conv => conv.id === currentConversationId)
        );
        if (hasCurrentConversation) {
          const currentGroup = data.find(group =>
            group.conversations.some(conv => conv.id === currentConversationId)
          );
          if (currentGroup?.document?.id) {
            setExpandedDocuments(prev => new Set([...prev, currentGroup.document!.id]));
          }
        }
      }
    } catch (error) {
      console.error('Error refreshing conversations: ', error)
    } finally {
      setIsLoading(false);
    }
  }, [userId, currentConversationId]);

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
    if (confirm("Are you sure you want to delete this conversation?")) {
      setDeleting(conversationId);

      try {
        const response = await conversationApi.delete(conversationId);

        if (!response.ok) {
          throw new Error('Failed to delete conversation');
        }

        // Remove conversation from the group
        setDocumentGroups(prev => prev.map(group => {
          if (group.document?.id === documentId) {
            return {
              ...group,
              conversations: group.conversations.filter(c => c.id !== conversationId)
            };
          }
          return group;
        }).filter(group => group.conversations.length > 0 || group.document)); // Remove empty groups

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

  const handleCreateNewConversation = async (e: React.MouseEvent, documentId: string) => {
    e.stopPropagation();
    e.preventDefault();
    if (!onCreateNewConversation) {
      console.warn('onCreateNewConversation prop not provided');
      return;
    }

    setCreatingConversation(documentId);
    // Expand the document first so the new conversation will be visible
    setExpandedDocuments(prev => new Set([...prev, documentId]));

    try {
      const newConv = await onCreateNewConversation(documentId);

      if (newConv && typeof newConv === 'object') {
        // Manually add the new conversation to the list
        // This avoids potential race conditions with refreshConversations returning stale data
        addNewConversation({
          id: newConv.id,
          documentId: documentId,
          title: newConv.title || 'New Chat',
          updatedAt: newConv.updated_at || new Date().toISOString()
        });
      } else {
        // Fallback if no conversation object returned
        await refreshConversations();
      }
    } catch (error) {
      console.error('Error creating new conversation: ', error);
      alert('Failed to create new conversation');
    } finally {
      setCreatingConversation(null);
    }
  }

  const toggleDocumentExpansion = (documentId: string) => {
    setExpandedDocuments(prev => {
      const newSet = new Set(prev);
      if (newSet.has(documentId)) {
        newSet.delete(documentId);
      } else {
        newSet.add(documentId);
      }
      return newSet;
    });
  };

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
          className="no-select no-tap-highlight p-2 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-all duration-200 active:scale-95 min-w-[44px] min-h-[44px] flex items-center justify-center"
          aria-label={isOpen ? "Collapse sidebar" : "Expand sidebar"}
        >
          {isOpen ? <ChevronLeft size={18} /> : <ChevronRight size={18} />}
        </button>
      </div>

      {/* New Chat Action */}
      <div className={`px-3 mb-6 ${isOpen ? '' : 'flex justify-center'}`}>
        <button
          onClick={handleNavigateToDashboard}
          className={`no-select no-tap-highlight group flex items-center gap-3 bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 text-white rounded-xl shadow-lg shadow-blue-900/20 transition-all duration-300 hover:shadow-blue-900/40 hover:-translate-y-0.5 ${
            isOpen ? 'w-full py-3 px-4 min-h-[44px]' : 'w-10 h-10 justify-center p-0 min-w-[44px] min-h-[44px]'
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
            Documents
          </div>
        )}

        {isLoading ? (
          <div className="flex justify-center items-center h-20">
            <div className="animate-spin rounded-full h-6 w-6 border-2 border-slate-600 border-t-blue-500"></div>
          </div>
        ) : documentGroups.length === 0 ? (
          <div className={`text-center p-4 text-slate-500 text-sm ${!isOpen && 'hidden'}`}>
            <div className="mb-2 inline-flex p-3 rounded-full bg-slate-800/50">
              <MessageSquare size={20} />
            </div>
            <p>No documents yet</p>
          </div>
        ) : (
          <ul className="space-y-1">
            {documentGroups.map((group) => {
              if (!group.document) return null;

              const documentId = group.document.id;
              const isExpanded = expandedDocuments.has(documentId);
              const conversationCount = group.conversations.length;
              const hasCurrentConversation = group.conversations.some(c => c.id === currentConversationId);

              return (
                <li key={documentId} className="group/document">
                  {/* Document Header */}
                  <div className="rounded-xl overflow-hidden">
                    <div className={`w-full rounded-xl transition-all duration-200 flex items-center justify-between group/item ${
                        hasCurrentConversation && !isExpanded
                          ? 'bg-slate-800/50 text-slate-200'
                          : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                      } ${isOpen ? 'p-3' : 'p-3 justify-center'}`}>
                      <button
                        onClick={() => toggleDocumentExpansion(documentId)}
                        className="no-select no-tap-highlight flex items-center gap-3 flex-1 min-w-0 text-left min-h-[44px]"
                      >
                        <div className="relative flex-shrink-0">
                          <FileText size={18} className={hasCurrentConversation ? 'text-blue-400' : 'text-slate-500 group-hover/item:text-slate-400'} />
                          {hasCurrentConversation && (
                            <span className="absolute -right-1 -top-1 w-2 h-2 rounded-full bg-blue-500 ring-2 ring-[#0f172a]" />
                          )}
                        </div>

                        {isOpen && (
                          <div className="overflow-hidden flex-1 min-w-0">
                            <div className="truncate font-medium text-sm leading-tight mb-0.5">
                              {group.document.title}
                            </div>
                            <div className="flex items-center gap-2 text-[10px] text-slate-500">
                              <span>{conversationCount} {conversationCount === 1 ? 'chat' : 'chats'}</span>
                            </div>
                          </div>
                        )}
                      </button>

                      {isOpen && (
                        <div className="flex items-center gap-1 flex-shrink-0">
                          {/* New Chat Button */}
                          <button
                            onClick={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              handleCreateNewConversation(e, documentId);
                            }}
                            disabled={creatingConversation === documentId}
                            className={`no-select no-tap-highlight p-1.5 rounded-lg transition-all min-w-[44px] min-h-[44px] flex items-center justify-center ${
                              creatingConversation === documentId
                                ? 'text-blue-400 opacity-100 cursor-not-allowed'
                                : 'text-slate-500 hover:bg-blue-500/10 hover:text-blue-400 opacity-70 group-hover/document:opacity-100'
                            }`}
                            title="New Chat"
                          >
                            {creatingConversation === documentId ? (
                              <div className="h-3.5 w-3.5 border-2 border-t-blue-500 border-transparent rounded-full animate-spin" />
                            ) : (
                              <Plus size={14} />
                            )}
                          </button>
                          {/* Expand/Collapse Button */}
                          <button
                            onClick={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              toggleDocumentExpansion(documentId);
                            }}
                            className="no-select no-tap-highlight p-1 rounded-lg hover:bg-slate-700/50 transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
                            title={isExpanded ? "Collapse" : "Expand"}
                          >
                            {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                          </button>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Conversations List (when expanded) */}
                  {isOpen && isExpanded && (
                    <div className="ml-4 mt-1 space-y-1 border-l border-slate-700/50 pl-3 pb-1">
                      {group.conversations.map((conversation) => (
                        <div key={conversation.id} className="group/conversation relative">
                          <button
                            onClick={() => onSelectConversation(conversation.id, documentId)}
                            className={`no-select no-tap-highlight w-full text-left rounded-lg transition-all duration-200 flex items-center group/item min-h-[44px] ${
                              currentConversationId === conversation.id
                                ? 'bg-slate-800 text-white shadow-sm ring-1 ring-slate-700/50'
                                : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                            } p-2.5`}
                          >
                            <div className="relative flex-shrink-0 mr-2">
                              <MessageSquare size={14} className={currentConversationId === conversation.id ? 'text-blue-400' : 'text-slate-500 group-hover/item:text-slate-400'} />
                              {currentConversationId === conversation.id && (
                                <span className="absolute -right-0.5 -top-0.5 w-1.5 h-1.5 rounded-full bg-blue-500 ring-1 ring-[#0f172a]" />
                              )}
                            </div>

                            <div className="overflow-hidden flex-1 min-w-0">
                              <div className="truncate font-medium text-xs leading-tight mb-0.5">
                                {conversation.title || 'New Chat'}
                              </div>
                              <div className="text-[10px] text-slate-500">
                                {formatDate(conversation.updated_at)}
                              </div>
                            </div>
                          </button>

                          {/* Delete Button */}
                          <div className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover/conversation:opacity-100 transition-opacity">
                            <button
                              onClick={(e) => handleDelete(e, conversation.id, documentId)}
                              className={`no-select no-tap-highlight p-1 rounded-lg hover:bg-red-500/10 hover:text-red-400 transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center ${
                                deleting === conversation.id ? 'text-red-400 opacity-100' : 'text-slate-500'
                              }`}
                              title="Delete conversation"
                              disabled={deleting === conversation.id}
                            >
                              {deleting === conversation.id ? (
                                <div className="h-3 w-3 border-2 border-t-red-500 border-transparent rounded-full animate-spin" />
                              ) : (
                                <Trash2 size={12} />
                              )}
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* Footer User Menu */}
      <div className="p-3 border-t border-slate-800 bg-[#0f172a]">
        <div className="relative">
          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className={`no-tap-highlight w-full flex items-center rounded-xl hover:bg-slate-800 transition-colors min-h-[44px] ${
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
                  className="no-select no-tap-highlight flex items-center px-3 py-2 text-sm text-slate-300 hover:bg-slate-700/50 hover:text-white rounded-lg transition-colors gap-2 min-h-[44px]"
                >
                  <Settings size={16} className="text-slate-400" />
                  <span>Settings</span>
                </Link>
                <div className="h-px bg-slate-700/50 my-1" />
                <button
                  onClick={handleLogout}
                  className="no-select no-tap-highlight w-full flex items-center px-3 py-2 text-sm text-red-400 hover:bg-red-500/10 hover:text-red-300 rounded-lg transition-colors gap-2 min-h-[44px]"
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

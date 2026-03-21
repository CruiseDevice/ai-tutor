// src/components/ChatSidebar.tsx
import Link from "next/link";
import { useRouter } from "next/navigation";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState, useCallback } from "react";
import { authApi } from "@/lib/api-client";

// Store imports for Zustand migration
import { useAuthStore } from "@/stores/authStore";
import { useDocumentsStore, selectDocumentGroups, selectIsLoadingDocs, selectExpandedDocuments } from "@/stores/documentsStore";
import { useChatStore } from "@/stores/chatStore";

interface Conversation {
  id: string;
  documentId: string;
  title: string;
  updatedAt: string;
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type -- Component reads from store, no props needed
interface ChatSidebarProps {
  // No props needed - reads from store
}

export interface ChatSidebarRef {
  addNewConversation: (newConversation: Conversation) => void;
  refreshConversations: () => void;
}

const ChatSidebar = forwardRef<ChatSidebarRef, ChatSidebarProps>(({}, ref) => {
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(true); // Default to open for better UX
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // =====================================================
  // STORE HOOKS - always use store now
  // =====================================================
  const storeUserId = useAuthStore((s) => s.userId);
  const storeUserEmail = useAuthStore((s) => s.userEmail);
  const storeDocumentGroups = useDocumentsStore(selectDocumentGroups);
  const storeIsLoading = useDocumentsStore(selectIsLoadingDocs);
  const storeConversationId = useChatStore((s) => s.conversationId);
  const storeLoadConversation = useChatStore((s) => s.loadConversation);
  const storeClearChat = useChatStore((s) => s.clearChat);
  const storeDeleteConversation = useDocumentsStore((s) => s.deleteConversation);
  const storeCreateConversation = useDocumentsStore((s) => s.createConversation);
  const storeToggleExpanded = useDocumentsStore((s) => s.toggleExpanded);
  const storeExpandedDocuments = useDocumentsStore(selectExpandedDocuments);
  const storeDeletingConversationId = useDocumentsStore((s) => s.deletingConversationId);
  const storeCreatingConversationDocId = useDocumentsStore((s) => s.creatingConversationDocId);
  const storeFetchDocumentGroups = useDocumentsStore((s) => s.fetchDocumentGroups);

  // Load conversations on mount
  useEffect(() => {
    storeFetchDocumentGroups();
  }, [storeFetchDocumentGroups]);

  // Expose ref methods for backward compatibility (delegate to store)
  useImperativeHandle(ref, () => ({
    addNewConversation: () => {
      // Store handles this automatically now - no-op for compatibility
    },
    refreshConversations: () => {
      storeFetchDocumentGroups();
    }
  }));

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

  const handleSelectConversation = useCallback((conversationId: string, documentId: string) => {
    // documentId is passed for consistency but not currently used
    void documentId;
    storeLoadConversation(conversationId);
  }, [storeLoadConversation]);

  const handleDelete = useCallback(async(e: React.MouseEvent, conversationId: string, documentId: string) => {
    e.stopPropagation();
    if (confirm("Are you sure you want to delete this conversation?")) {
      const wasCurrentConversation = conversationId === storeConversationId;

      await storeDeleteConversation(conversationId, documentId);
      // Refresh document groups to update UI (handles removing empty document groups)
      await storeFetchDocumentGroups();

      // Clear the PDF viewer if we deleted the current conversation
      if (wasCurrentConversation) {
        storeClearChat();
      }
    }
  }, [storeDeleteConversation, storeFetchDocumentGroups, storeConversationId, storeClearChat]);

  const handleCreateNewConversation = useCallback(async (e: React.MouseEvent, documentId: string) => {
    e.stopPropagation();
    e.preventDefault();
    await storeCreateConversation(documentId);
  }, [storeCreateConversation]);

  const toggleDocumentExpansion = useCallback((documentId: string) => {
    storeToggleExpanded(documentId);
  }, [storeToggleExpanded]);

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
      className={`h-full bg-sidebar-bg text-ink border-r-2 border-ink flex flex-col relative z-20 font-serif brutalist-texture sidebar-collapse ${
        !isOpen ? 'collapsed' : ''
      }`}
    >
      {/* =====================================================
          [001] HEADER - TUTOR.AI branding
          ===================================================== */}
      <div className="border-b-2 border-ink">
        <div className={`flex items-center p-4 ${isOpen ? 'justify-between' : 'justify-center'}`}>
          {isOpen && (
            <div className="flex items-center gap-3 overflow-hidden fade-content">
              <h2 className="font-mono font-bold text-lg uppercase tracking-tight">
                TUTOR<span className="text-accent">.AI</span>
              </h2>
              <span className="font-mono text-xs text-accent ml-auto fade-content">[001]</span>
            </div>
          )}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="no-select brutalist-shadow-sm p-2 border border-ink hover:bg-ink hover:text-paper transition-all duration-150 min-w-[44px] min-h-[44px] flex items-center justify-center font-mono text-sm"
            aria-label={isOpen ? "Collapse sidebar" : "Expand sidebar"}
          >
            <span className={`transition-transform duration-200 ${isOpen ? 'rotate-0' : 'rotate-180'}`}>
              {isOpen ? '[◀]' : '[▶]'}
            </span>
          </button>
        </div>
      </div>

      {/* =====================================================
          [NEW STUDY SESSION] button
          ===================================================== */}
      <div className={`p-4 border-b-2 border-ink ${isOpen ? '' : 'flex justify-center'}`}>
        <button
          onClick={handleNavigateToDashboard}
          className={`brutalist-button brutalist-button-primary brutalist-shadow w-full py-3 px-4 min-h-[44px] font-mono text-sm uppercase ${
            isOpen ? '' : 'w-10 justify-center p-0 min-w-[44px]'
          }`}
        >
          <span className={isOpen ? 'fade-content' : ''}>{isOpen ? '[NEW STUDY SESSION]' : '[+]'}</span>
        </button>
      </div>

      {/* =====================================================
          [002] DOCUMENTS section
          ===================================================== */}
      <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-200">
        {isOpen && (
          <div className="flex items-center gap-2 px-4 py-3 border-b-2 border-ink fade-content">
            <span className="font-mono text-xs text-accent">[002]</span>
            <span className="font-mono text-xs uppercase tracking-wider">Documents</span>
          </div>
        )}

        {storeIsLoading ? (
          <div className="flex justify-center items-center h-20">
            <div className="font-mono text-xs text-subtle animate-pulse">[LOADING...]</div>
          </div>
        ) : storeDocumentGroups.length === 0 ? (
          <div className={`text-center p-8 ${!isOpen && 'hidden'}`}>
            <div className="font-mono text-subtle/40 text-4xl mb-4 animate-float fade-content">[∅]</div>
            <p className="font-mono text-xs text-subtle mb-2 fade-content">[EMPTY LIBRARY]</p>
            <p className="font-serif text-xs text-subtle/70 leading-relaxed fade-content">
              Upload a PDF to begin your study session
            </p>
            <div className="mt-4 font-mono text-[10px] text-subtle/50 fade-content">
              └ <span className="text-accent">[NEW STUDY SESSION]</span> to get started
            </div>
          </div>
        ) : (
          <ul className="border-t-2 border-ink">
            {storeDocumentGroups
              .filter((group) => group.document != null)
              .map((group) => {
              const documentId = group.document!.id;
              const isExpanded = storeExpandedDocuments.has(documentId);
              const conversationCount = group.conversations.length;
              const hasCurrentConversation = group.conversations.some(c => c.id === storeConversationId);

              return (
                <li key={documentId} className="group/document border-b border-ink last:border-b-0">
                  {/* Document Header */}
                  <div
                    className={`w-full transition-colors duration-150 flex items-center justify-between group/item ${
                        hasCurrentConversation && !isExpanded
                          ? 'bg-accent/10'
                          : 'hover:bg-accent/5'
                      } ${isOpen ? 'p-3' : 'p-3 justify-center'}`}
                  >
                    <button
                      onClick={() => toggleDocumentExpansion(documentId)}
                      className="no-select flex items-center gap-3 flex-1 min-w-0 text-left min-h-[44px]"
                    >
                      <div className="relative flex-shrink-0">
                        <span className={`font-mono text-xs ${hasCurrentConversation ? 'text-accent' : 'text-subtle group-hover/item:text-ink'}`}>
                          [PDF]
                        </span>
                        {hasCurrentConversation && (
                          <span className="absolute -right-1 -top-0.5 w-1.5 h-1.5 bg-accent" />
                        )}
                      </div>

                      {isOpen && (
                        <div className="overflow-hidden flex-1 min-w-0 fade-content">
                          <div className="truncate font-medium text-sm leading-tight mb-0.5">
                            {group.document.title}
                          </div>
                          <div className="font-mono text-[10px] text-subtle">
                            └ [{conversationCount} chat{conversationCount !== 1 ? 's' : ''}]
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
                          disabled={storeCreatingConversationDocId === documentId}
                          className={`no-select brutalist-shadow-sm font-mono text-xs px-2 py-1 border border-ink transition-all min-w-[44px] min-h-[44px] flex items-center justify-center ${
                            storeCreatingConversationDocId === documentId
                              ? 'text-accent opacity-100 cursor-not-allowed'
                              : 'text-subtle hover:bg-ink hover:text-paper opacity-70 group-hover/document:opacity-100'
                          }`}
                          title="New Chat"
                        >
                          {storeCreatingConversationDocId === documentId ? (
                            <span className="animate-pulse">[...]</span>
                          ) : (
                            '[+]'
                          )}
                        </button>
                        {/* Expand/Collapse Button */}
                        <button
                          onClick={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            toggleDocumentExpansion(documentId);
                          }}
                          className="no-select brutalist-shadow-sm font-mono text-xs px-2 py-1 border border-ink hover:bg-ink hover:text-paper transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center"
                          title={isExpanded ? "Collapse" : "Expand"}
                        >
                          {isExpanded ? '[▲]' : '[▼]'}
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Conversations List (when expanded) */}
                  {isOpen && isExpanded && (
                    <div className="ml-4 mt-1 space-y-1 border-l-2 border-ink pl-3 pb-2 bg-sidebar-bg/50">
                      {group.conversations.map((conversation, index) => (
                        <div
                          key={`${documentId}-${conversation.id}`}
                          className="group/conversation relative animate-slide-in-stagger"
                          style={{ animationDelay: `${index * 60}ms` }}
                        >
                          <button
                            onClick={() => handleSelectConversation(conversation.id, documentId)}
                            className={`no-select brutalist-shadow-sm w-full text-left border transition-all duration-150 flex items-center group/item min-h-[44px] p-2.5 ${
                              storeConversationId === conversation.id
                                ? 'bg-ink text-paper border-ink'
                                : 'bg-paper text-ink border-subtle hover:bg-ink hover:text-paper hover:border-ink'
                            }`}
                          >
                            <div className="relative flex-shrink-0 mr-2">
                              <span className={`font-mono text-xs ${
                                storeConversationId === conversation.id ? 'text-paper' : 'text-subtle group-hover/item:text-paper'
                              }`}>
                                [§]
                              </span>
                            </div>

                            <div className="overflow-hidden flex-1 min-w-0 fade-content">
                              <div className="truncate font-medium text-xs leading-tight mb-0.5">
                                {conversation.title || 'New Chat'}
                              </div>
                              <div className={`font-mono text-[10px] ${
                                storeConversationId === conversation.id ? 'text-subtle/70' : 'text-subtle'
                              }`}>
                                {formatDate(conversation.updated_at)}
                              </div>
                            </div>
                          </button>

                          {/* Delete Button */}
                          <div className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover/conversation:opacity-100 transition-opacity">
                            <button
                              onClick={(e) => handleDelete(e, conversation.id, documentId)}
                              className={`no-select brutalist-shadow-sm font-mono text-xs px-2 py-1 border border-ink hover:bg-[var(--danger)] hover:border-[var(--danger)] hover:text-[var(--paper)] transition-colors min-w-[44px] min-h-[44px] flex items-center justify-center ${
                                storeDeletingConversationId === conversation.id ? 'text-[var(--danger)] opacity-100' : 'text-subtle'
                              }`}
                              title="Delete conversation"
                              disabled={storeDeletingConversationId === conversation.id}
                            >
                              {storeDeletingConversationId === conversation.id ? (
                                <span className="animate-pulse">[×]</span>
                              ) : (
                                '[×]'
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

      {/* =====================================================
          FOOTER - User section with [PRO PLAN] badge
          ===================================================== */}
      <div className="border-t-2 border-ink p-4">
        <div className="relative">
          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className={`no-tap-highlight brutalist-shadow-sm w-full flex items-center border hover:bg-ink hover:text-paper transition-colors min-h-[44px] ${
              isOpen ? 'p-3 justify-between' : 'p-3 justify-center'
            } ${isMenuOpen ? 'bg-ink text-paper' : 'border-ink'}`}
          >
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-ink text-paper flex items-center justify-center font-mono text-xs border border-ink">
                {storeUserId ? storeUserId.substring(0, 2).toUpperCase() : 'U'}
              </div>
              {isOpen && (
                <div className="text-left fade-content">
                  <div className="font-serif text-xs truncate w-32">
                    {storeUserEmail || storeUserId || 'User'}
                  </div>
                </div>
              )}
            </div>
            {isOpen && (
              <div className="flex items-center gap-2 fade-content">
                <span className="font-mono text-xs px-2 py-1 border border-ink text-subtle">[PRO PLAN]</span>
                <span className={`font-mono text-xs transition-transform duration-200 ${isMenuOpen ? 'rotate-180' : ''}`}>[▼]</span>
              </div>
            )}
          </button>

          {/* Popup Menu - Brutalist Style */}
          {isMenuOpen && (
            <div
              ref={menuRef}
              className={`absolute bottom-full left-0 bg-paper border-2 border-ink shadow-none overflow-hidden z-50 mb-2 ${
                isOpen ? 'w-full min-w-[200px]' : 'left-10 w-48'
              }`}
            >
              <div className="border-b border-ink">
                <Link
                  href="/settings"
                  className="no-select brutalist-shadow-sm flex items-center px-4 py-3 font-mono text-xs text-ink hover:bg-ink hover:text-paper transition-colors gap-3 min-h-[44px] border-b border-ink"
                >
                  <span>[⚙]</span>
                  <span>Settings</span>
                </Link>
              </div>
              <div>
                <button
                  onClick={handleLogout}
                  className="no-select w-full flex items-center px-4 py-3 font-mono text-xs text-[var(--danger)] hover:bg-[var(--danger)] hover:text-[var(--paper)] transition-colors gap-3 min-h-[44px]"
                >
                  <span>[←]</span>
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

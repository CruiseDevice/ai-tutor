// src/components/ChatSidebar.tsx
import Link from "next/link";
import { useRouter } from "next/navigation";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState, useCallback } from "react";
import { PanelLeftClose, PanelLeftOpen, Plus, ChevronDown, ChevronUp, ChevronRight, X, Settings, LogOut, FileText, MessageSquare } from "lucide-react";
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
      className={`h-full bg-desk text-ink border-r border-hair flex flex-col relative z-20 font-serif sidebar-collapse ${
        !isOpen ? 'collapsed' : ''
      }`}
    >
      {/* =====================================================
          HEADER — TUTOR.AI wordmark
          ===================================================== */}
      <div className="border-b border-hair">
        <div className={`flex items-center p-4 ${isOpen ? 'justify-between' : 'justify-center'}`}>
          {isOpen && (
            <h2 className="font-serif font-semibold text-xl tracking-tight fade-content">
              TUTOR<span className="text-accent">.</span>AI
            </h2>
          )}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="no-select text-subtle hover:text-ink hover:bg-paper rounded-sm p-2 transition-colors ease-desk min-w-[44px] min-h-[44px] flex items-center justify-center"
            aria-label={isOpen ? "Collapse sidebar" : "Expand sidebar"}
          >
            {isOpen ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeftOpen className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {/* =====================================================
          New study session button
          ===================================================== */}
      <div className={`p-3 border-b border-hair ${isOpen ? '' : 'flex justify-center'}`}>
        <button
          onClick={handleNavigateToDashboard}
          className={`btn btn-primary w-full ${isOpen ? '' : '!w-11 !p-0 justify-center'}`}
        >
          <Plus className="h-4 w-4" />
          {isOpen && <span className="fade-content">New study session</span>}
        </button>
      </div>

      {/* =====================================================
          DOCUMENTS section
          ===================================================== */}
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        {isOpen && (
          <div className="px-4 pt-4 pb-2 fade-content">
            <span className="font-mono text-[11px] uppercase tracking-wider text-faint">Documents</span>
          </div>
        )}

        {storeIsLoading ? (
          <div className="flex justify-center items-center h-20">
            <div className="text-xs text-faint animate-pulse fade-content">Loading…</div>
          </div>
        ) : storeDocumentGroups.length === 0 ? (
          <div className={`text-center p-8 ${!isOpen && 'hidden'}`}>
            <FileText className="h-7 w-7 text-faint mx-auto mb-3 fade-content" />
            <p className="text-sm text-subtle mb-1 fade-content">Your library is empty</p>
            <p className="text-xs text-faint leading-relaxed fade-content">
              Upload a PDF to begin your study session
            </p>
          </div>
        ) : (
          <ul className="px-2">
            {storeDocumentGroups
              .filter((group) => group.document != null)
              .map((group) => {
              const documentId = group.document!.id;
              const isExpanded = storeExpandedDocuments.has(documentId);
              const conversationCount = group.conversations.length;
              const hasCurrentConversation = group.conversations.some(c => c.id === storeConversationId);

              return (
                <li key={documentId} className="group/document">
                  {/* Document Header */}
                  <div
                    className={`w-full rounded-sm transition-colors ease-desk flex items-center justify-between group/item ${
                      hasCurrentConversation && !isExpanded
                        ? 'bg-accent-soft'
                        : 'hover:bg-paper'
                    } ${isOpen ? 'p-2.5' : 'p-2.5 justify-center'}`}
                  >
                    <button
                      onClick={() => toggleDocumentExpansion(documentId)}
                      className="no-select flex items-center gap-2.5 flex-1 min-w-0 text-left min-h-[44px]"
                    >
                      <FileText className={`h-4 w-4 flex-shrink-0 ${hasCurrentConversation ? 'text-accent' : 'text-faint group-hover/item:text-subtle'}`} />

                      {isOpen && (
                        <div className="overflow-hidden flex-1 min-w-0 fade-content">
                          <div className="truncate text-sm leading-tight text-ink">
                            {group.document.title}
                          </div>
                          <div className="font-mono text-[10px] text-faint mt-0.5">
                            {conversationCount} chat{conversationCount !== 1 ? 's' : ''}
                          </div>
                        </div>
                      )}
                    </button>

                    {isOpen && (
                      <div className="flex items-center gap-0.5 flex-shrink-0">
                        {/* New Chat Button */}
                        <button
                          onClick={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            handleCreateNewConversation(e, documentId);
                          }}
                          disabled={storeCreatingConversationDocId === documentId}
                          className="no-select text-subtle hover:text-accent hover:bg-paper rounded-xs p-1.5 transition-colors ease-desk min-w-[44px] min-h-[44px] flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
                          title="New chat"
                          aria-label="New chat"
                        >
                          {storeCreatingConversationDocId === documentId ? (
                            <span className="font-mono text-[10px] animate-pulse">…</span>
                          ) : (
                            <Plus className="h-3.5 w-3.5" />
                          )}
                        </button>
                        {/* Expand/Collapse Button */}
                        <button
                          onClick={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            toggleDocumentExpansion(documentId);
                          }}
                          className="no-select text-subtle hover:text-ink hover:bg-paper rounded-xs p-1.5 transition-colors ease-desk min-w-[44px] min-h-[44px] flex items-center justify-center"
                          title={isExpanded ? "Collapse" : "Expand"}
                          aria-label={isExpanded ? "Collapse document" : "Expand document"}
                        >
                          {isExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Conversations List (when expanded) */}
                  {isOpen && isExpanded && (
                    <div className="ml-4 mt-0.5 mb-1 space-y-px border-l border-hair pl-2">
                      {group.conversations.map((conversation, index) => (
                        <div
                          key={`${documentId}-${conversation.id}`}
                          className="group/conversation relative animate-slide-in-stagger"
                          style={{ animationDelay: `${index * 60}ms` }}
                        >
                          <button
                            onClick={() => handleSelectConversation(conversation.id, documentId)}
                            className={`no-select w-full text-left rounded-sm transition-colors ease-desk flex items-center gap-2 min-h-[40px] p-2 ${
                              storeConversationId === conversation.id
                                ? 'bg-accent-soft text-ink'
                                : 'text-subtle hover:bg-paper hover:text-ink'
                            }`}
                          >
                            <MessageSquare className={`h-3 w-3 flex-shrink-0 ${storeConversationId === conversation.id ? 'text-accent' : 'text-faint'}`} />

                            <div className="overflow-hidden flex-1 min-w-0 fade-content">
                              <div className={`truncate text-xs leading-tight ${storeConversationId === conversation.id ? 'text-ink' : ''}`}>
                                {conversation.title || 'New chat'}
                              </div>
                              <div className="font-mono text-[10px] text-faint mt-0.5">
                                {formatDate(conversation.updated_at)}
                              </div>
                            </div>
                          </button>

                          {/* Delete Button */}
                          <div className="absolute right-1 top-1/2 -translate-y-1/2 opacity-0 group-hover/conversation:opacity-100 transition-opacity">
                            <button
                              onClick={(e) => handleDelete(e, conversation.id, documentId)}
                              className={`no-select hover:bg-accent-soft rounded-xs p-1.5 transition-colors ease-desk min-w-[44px] min-h-[44px] flex items-center justify-center ${
                                storeDeletingConversationId === conversation.id ? 'text-danger opacity-100' : 'text-faint hover:text-danger'
                              }`}
                              title="Delete conversation"
                              aria-label="Delete conversation"
                              disabled={storeDeletingConversationId === conversation.id}
                            >
                              {storeDeletingConversationId === conversation.id ? (
                                <span className="font-mono text-[10px] animate-pulse">…</span>
                              ) : (
                                <X className="h-3.5 w-3.5" />
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
          FOOTER — user section (no fake PRO PLAN badge)
          ===================================================== */}
      <div className="border-t border-hair p-3">
        <div className="relative">
          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className={`no-tap-highlight w-full flex items-center gap-2.5 rounded-sm transition-colors ease-desk min-h-[44px] ${
              isOpen ? 'p-2 justify-between hover:bg-paper' : 'p-2 justify-center hover:bg-paper'
            } ${isMenuOpen ? 'bg-paper' : ''}`}
            aria-label="Open user menu"
            aria-expanded={isMenuOpen}
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="tutor-avatar !w-8 !h-8 !text-xs">
                {storeUserId ? storeUserId.substring(0, 2).toUpperCase() : 'U'}
              </div>
              {isOpen && (
                <div className="text-left fade-content min-w-0">
                  <div className="text-xs text-ink truncate w-32">
                    {storeUserEmail || storeUserId || 'User'}
                  </div>
                </div>
              )}
            </div>
            {isOpen && (
              <ChevronRight className={`h-3.5 w-3.5 text-faint transition-transform duration-200 fade-content ${isMenuOpen ? 'rotate-90' : ''}`} />
            )}
          </button>

          {/* Popup Menu */}
          {isMenuOpen && (
            <div
              ref={menuRef}
              className="absolute bottom-full left-0 right-0 mb-2 bg-surface border border-hair rounded-sm shadow-card overflow-hidden z-50 model-menu-enter"
            >
              <div className="border-b border-hair-soft">
                <Link
                  href="/settings"
                  className="no-select flex items-center px-3 py-2.5 text-sm text-ink hover:bg-desk transition-colors ease-desk gap-2.5 min-h-[44px]"
                >
                  <Settings className="h-4 w-4 text-subtle" />
                  <span>Settings</span>
                </Link>
              </div>
              <div>
                <button
                  onClick={handleLogout}
                  className="no-select w-full flex items-center px-3 py-2.5 text-sm text-danger hover:bg-accent-soft transition-colors ease-desk gap-2.5 min-h-[44px]"
                >
                  <LogOut className="h-4 w-4" />
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

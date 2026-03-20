// src/stores/utils/crossTabSync.ts

/**
 * Cross-tab synchronization for conversation state.
 *
 * Uses localStorage's storage event to broadcast conversation changes
 * between browser tabs. Only syncs critical state (conversationId, documentId)
 * to avoid unnecessary complexity.
 *
 * NOTE: storage events only fire in OTHER tabs, not the tab that made the change.
 */

const STORAGE_KEY = 'chat_current_conversation';

export interface CrossTabState {
  conversationId: string | null;
  documentId: string | null;
  timestamp: number;
}

/**
 * Listen for conversation changes from other tabs.
 * Calls onLoadConversation when a different conversation is detected.
 */
export function listenForCrossTabChanges(
  currentConversationId: string | null,
  onLoadConversation: (conversationId: string) => void
): () => void {
  const handler = (e: StorageEvent) => {
    if (e.key === STORAGE_KEY && e.newValue) {
      try {
        const data: CrossTabState = JSON.parse(e.newValue);

        // Only load if it's a different conversation (prevent infinite loop)
        if (data.conversationId && data.conversationId !== currentConversationId) {
          console.log('[CrossTab] Loading conversation from another tab:', data.conversationId);
          onLoadConversation(data.conversationId);
        }
      } catch (error) {
        console.error('[CrossTab] Failed to parse storage event:', error);
      }
    }
  };

  window.addEventListener('storage', handler);

  // Return cleanup function
  return () => window.removeEventListener('storage', handler);
}

/**
 * Broadcast conversation change to other tabs.
 * Call this when the user selects or creates a new conversation.
 */
export function broadcastConversationChange(
  conversationId: string | null,
  documentId: string | null
): void {
  if (typeof window === 'undefined') return;

  const state: CrossTabState = {
    conversationId,
    documentId,
    timestamp: Date.now(),
  };

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (error) {
    console.error('[CrossTab] Failed to broadcast conversation change:', error);
  }
}

/**
 * Get the current conversation state from localStorage.
 * Useful for initializing store on page load.
 */
export function getCrossTabState(): CrossTabState | null {
  if (typeof window === 'undefined') return null;

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (error) {
    console.error('[CrossTab] Failed to read state:', error);
  }

  return null;
}

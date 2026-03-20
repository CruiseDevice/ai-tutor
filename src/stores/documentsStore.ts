// src/stores/documentsStore.ts
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { conversationApi } from '../lib/api-client';

interface Document {
  id: string;
  title: string;
  url: string;
}

interface Conversation {
  id: string;
  user_id: string;
  document_id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
}

interface DocumentGroup {
  document: Document | null;
  conversations: Conversation[];
}

interface DocumentsState {
  documentGroups: DocumentGroup[];
  isLoading: boolean;
  expandedDocuments: Set<string>;
  deletingConversationId: string | null;
  creatingConversationDocId: string | null;

  // Actions
  setDocumentGroups: (groups: DocumentGroup[]) => void;
  addConversation: (docId: string, conversation: Conversation) => void;
  removeConversation: (conversationId: string, docId: string) => void;
  updateConversationTitle: (conversationId: string, title: string) => void;
  toggleExpanded: (docId: string) => void;
  setExpanded: (docId: string, expanded: boolean) => void;
  setDeleting: (id: string | null) => void;
  setCreating: (id: string | null) => void;
  setLoading: (loading: boolean) => void;

  // Async Actions
  fetchDocumentGroups: (groupByDocument?: boolean) => Promise<void>;
  createConversation: (documentId: string) => Promise<Conversation | null>;
  deleteConversation: (conversationId: string, docId: string) => Promise<boolean>;
}

export const useDocumentsStore = create<DocumentsState>()(
  devtools(
    (set, get) => ({
      documentGroups: [],
      isLoading: false,
      expandedDocuments: new Set(),
      deletingConversationId: null,
      creatingConversationDocId: null,

      // Synchronous Actions
      setDocumentGroups: (documentGroups) => set({ documentGroups }),

      addConversation: (docId, conversation) => set((state) => ({
        documentGroups: state.documentGroups.map((group) => {
          if (group.document?.id === docId) {
            const exists = group.conversations.some((c) => c.id === conversation.id);
            if (exists) return group;
            return {
              ...group,
              conversations: [conversation, ...group.conversations],
            };
          }
          return group;
        }),
      })),

      removeConversation: (conversationId, docId) => set((state) => ({
        documentGroups: state.documentGroups
          .map((group) => {
            if (group.document?.id === docId) {
              return {
                ...group,
                conversations: group.conversations.filter((c) => c.id !== conversationId),
              };
            }
            return group;
          })
          .filter((group) => group.conversations.length > 0 || group.document),
      })),

      updateConversationTitle: (conversationId, title) => set((state) => ({
        documentGroups: state.documentGroups.map((group) => ({
          ...group,
          conversations: group.conversations.map((conv) =>
            conv.id === conversationId ? { ...conv, title } : conv
          ),
        })),
      })),

      toggleExpanded: (docId) => set((state) => {
        const newSet = new Set(state.expandedDocuments);
        if (newSet.has(docId)) {
          newSet.delete(docId);
        } else {
          newSet.add(docId);
        }
        return { expandedDocuments: newSet };
      }),

      setExpanded: (docId, expanded) => set((state) => {
        const newSet = new Set(state.expandedDocuments);
        if (expanded) {
          newSet.add(docId);
        } else {
          newSet.delete(docId);
        }
        return { expandedDocuments: newSet };
      }),

      setDeleting: (deletingConversationId) => set({ deletingConversationId }),

      setCreating: (creatingConversationDocId) => set({ creatingConversationDocId }),

      setLoading: (isLoading) => set({ isLoading }),

      // Async Actions
      fetchDocumentGroups: async (groupByDocument = true) => {
        set({ isLoading: true });
        try {
          const response = await conversationApi.list(groupByDocument);
          if (!response.ok) throw new Error('Failed to fetch conversations');

          const data = await response.json() as DocumentGroup[];
          set({ documentGroups: data });
        } catch (error) {
          console.error('Error fetching document groups:', error);
        } finally {
          set({ isLoading: false });
        }
      },

      createConversation: async (documentId) => {
        set({ creatingConversationDocId: documentId });
        get().setExpanded(documentId, true);

        try {
          const response = await conversationApi.create(documentId);
          if (!response.ok) throw new Error('Failed to create conversation');

          const conversation = await response.json() as Conversation;
          get().addConversation(documentId, conversation);
          return conversation;
        } catch (error) {
          console.error('Error creating conversation:', error);
          return null;
        } finally {
          set({ creatingConversationDocId: null });
        }
      },

      deleteConversation: async (conversationId, docId) => {
        set({ deletingConversationId: conversationId });
        try {
          const response = await conversationApi.delete(conversationId);
          if (!response.ok) throw new Error('Failed to delete conversation');

          get().removeConversation(conversationId, docId);
          return true;
        } catch (error) {
          console.error('Error deleting conversation:', error);
          return false;
        } finally {
          set({ deletingConversationId: null });
        }
      },
    }),
    { name: 'DocumentsStore' }
  )
);

// Selectors
export const selectDocumentGroups = (state: DocumentsState) => state.documentGroups;
export const selectIsLoadingDocs = (state: DocumentsState) => state.isLoading;

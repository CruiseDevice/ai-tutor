// src/stores/chatStore.ts
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { conversationApi, chatApi, getPDFProxyUrl } from '../lib/api-client';
import { StreamBatcher } from './utils/StreamBatcher';
import { listenForCrossTabChanges, broadcastConversationChange } from './utils/crossTabSync';
import { useAnnotationsStore } from './annotationsStore';
import type { ChatMessage, WorkflowStep, AnnotationReference } from '../types';

interface ChatState {
  // State
  conversationId: string | null;
  messages: ChatMessage[];
  currentPDF: string;
  documentId: string;
  isLoading: boolean;
  error: string | null;
  workflowSteps: WorkflowStep[];
  showWorkflow: boolean;
  selectedModel: string;
  hasApiKey: boolean;
  useAgent: boolean;

  // Synchronous Actions
  setConversation: (id: string, docId: string, pdfUrl: string) => void;
  setCurrentPDF: (pdfUrl: string) => void;
  setMessages: (messages: ChatMessage[]) => void;
  addMessage: (message: ChatMessage) => void;
  updateMessage: (id: string, updates: Partial<ChatMessage>) => void;
  removeMessage: (id: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setWorkflow: (steps: WorkflowStep[]) => void;
  updateWorkflowStep: (node: string, status: WorkflowStep['status'], data?: Record<string, unknown>) => void;
  toggleWorkflow: (show: boolean) => void;
  setSelectedModel: (model: string) => void;
  setApiKeyStatus: (hasKey: boolean) => void;
  toggleAgent: () => void;
  clearChat: () => void;

  // Async Actions (thunks)
  loadConversation: (id: string) => Promise<void>;
  createConversation: (documentId: string) => Promise<void>;
  sendMessage: (content: string) => Promise<void>;
  deleteConversation: (id: string) => Promise<void>;
}

export const useChatStore = create<ChatState>()(
  devtools(
    (set, get) => ({
      // Initial State
      conversationId: null,
      messages: [],
      currentPDF: '',
      documentId: '',
      isLoading: false,
      error: null,
      workflowSteps: [],
      showWorkflow: false,
      selectedModel: 'gpt-5.1',
      hasApiKey: false,
      useAgent: true,

      // Synchronous Actions
      setConversation: (id, docId, pdfUrl) => {
        set({ conversationId: id, documentId: docId, currentPDF: pdfUrl });
        // Broadcast to other tabs
        broadcastConversationChange(id, docId);
      },

      setCurrentPDF: (pdfUrl) => set({ currentPDF: pdfUrl }),

      setMessages: (messages) => set({ messages }),

      addMessage: (message) => set((state) => ({
        messages: [...state.messages, message],
      })),

      updateMessage: (id, updates) => set((state) => ({
        messages: state.messages.map((msg) =>
          msg.id === id ? { ...msg, ...updates } : msg
        ),
      })),

      removeMessage: (id) => set((state) => ({
        messages: state.messages.filter((msg) => msg.id !== id),
      })),

      setLoading: (isLoading) => set({ isLoading }),

      setError: (error) => set({ error }),

      setWorkflow: (workflowSteps) => set({ workflowSteps }),

      updateWorkflowStep: (node, status, data) => set((state) => ({
        workflowSteps: state.workflowSteps.map((step) =>
          step.node === node ? { ...step, status, data } : step
        ),
      })),

      toggleWorkflow: (showWorkflow) => set({ showWorkflow }),

      setSelectedModel: (selectedModel) => set({ selectedModel }),

      setApiKeyStatus: (hasApiKey) => set({ hasApiKey }),

      toggleAgent: () => set((state) => ({ useAgent: !state.useAgent })),

      clearChat: () => set({
        conversationId: null,
        messages: [],
        currentPDF: '',
        documentId: '',
        error: null,
        workflowSteps: [],
        showWorkflow: false,
      }),

      // Async Actions
      loadConversation: async (id) => {
        set({ isLoading: true, error: null });
        try {
          const response = await conversationApi.get(id);
          if (!response.ok) throw new Error('Failed to load conversation');

          const data = await response.json();
          set({
            conversationId: id,
            messages: data.messages,
            documentId: data.document_id,
            currentPDF: getPDFProxyUrl(data.document_id),
          });
        } catch (error) {
          set({ error: error instanceof Error ? error.message : 'Failed to load conversation' });
        } finally {
          set({ isLoading: false });
        }
      },

      createConversation: async (documentId) => {
        set({ isLoading: true, error: null });
        try {
          const response = await conversationApi.create(documentId);
          if (!response.ok) throw new Error('Failed to create conversation');

          const data = await response.json();
          set({
            conversationId: data.id,
            documentId,
            currentPDF: getPDFProxyUrl(documentId),
            messages: [],
          });
        } catch (error) {
          set({ error: error instanceof Error ? error.message : 'Failed to create conversation' });
        } finally {
          set({ isLoading: false });
        }
      },

      sendMessage: async (content) => {
        const { conversationId, documentId, selectedModel, useAgent } = get();
        if (!conversationId || !documentId) return;

        // Add user message optimistically
        const userMsgId = `temp-user-${Date.now()}`;
        const assistantMsgId = `temp-assistant-${Date.now()}`;

        set((state) => ({
          messages: [
            ...state.messages,
            { id: userMsgId, role: 'user', content },
            { id: assistantMsgId, role: 'assistant', content: '' },
          ],
          isLoading: true,
          error: null,
          showWorkflow: useAgent,
          workflowSteps: useAgent ? [
            { node: 'understand_query', status: 'in_progress' },
            { node: 'retrieve_context', status: 'pending' },
            { node: 'generate_answer', status: 'pending' },
            { node: 'format_response', status: 'pending' },
          ] : [],
        }));

        // Create batcher for streaming updates
        const batcher = new StreamBatcher(
          (id, updatedContent) => {
            set((state) => ({
              messages: state.messages.map((msg) =>
                msg.id === id ? { ...msg, content: updatedContent } : msg
              ),
            }));
          }
        );

        try {
          await chatApi.sendMessageStream(
            conversationId,
            documentId,
            content,
            selectedModel,
            useAgent,
            // onChunk - batch updates for performance
            (chunk) => {
              batcher.addChunk(assistantMsgId, chunk);
            },
            // onStep
            (step) => {
              get().updateWorkflowStep(step.node, 'completed', step.data);
            },
            // onDone
            (data: unknown) => {
              batcher.complete(assistantMsgId);
              batcher.destroy();

              const response = data as { user_message?: ChatMessage; assistant_message?: ChatMessage; data?: { user_message?: ChatMessage; assistant_message?: ChatMessage } };
              const userData = response.user_message || response.data?.user_message;
              const assistantData = response.assistant_message || response.data?.assistant_message;

              if (userData && assistantData) {
                // Handle annotations from assistant response
                if (assistantData.annotations && assistantData.annotations.length > 0) {
                  useAnnotationsStore.getState().setAnnotations(assistantData.annotations);
                }

                set((state) => ({
                  messages: [
                    ...state.messages.filter(m => m.id !== userMsgId && m.id !== assistantMsgId),
                    userData,
                    assistantData,
                  ],
                  isLoading: false,
                  showWorkflow: false,
                }));
              }
            },
            // onError
            (error) => {
              batcher.destroy();
              set((state) => ({
                messages: state.messages.filter(m => m.id !== userMsgId && m.id !== assistantMsgId),
                error,
                isLoading: false,
                showWorkflow: false,
              }));
            }
          );
        } catch (error) {
          batcher.destroy();
          set((state) => ({
            messages: state.messages.filter(m => m.id !== userMsgId && m.id !== assistantMsgId),
            error: error instanceof Error ? error.message : 'Failed to send message',
            isLoading: false,
            showWorkflow: false,
          }));
        }
      },

      deleteConversation: async (id) => {
        try {
          const response = await conversationApi.delete(id);
          if (!response.ok) throw new Error('Failed to delete conversation');

          if (get().conversationId === id) {
            get().clearChat();
          }
        } catch (error) {
          set({ error: error instanceof Error ? error.message : 'Failed to delete conversation' });
        }
      },
    }),
    { name: 'ChatStore' }
  )
);

// Initialize cross-tab sync (only in browser)
if (typeof window !== 'undefined') {
  // Listen for changes from other tabs
  void listenForCrossTabChanges(
    useChatStore.getState().conversationId,
    (conversationId) => {
      useChatStore.getState().loadConversation(conversationId);
    }
  );

  // Cleanup on page unload (optional - browser handles this)
  // window.addEventListener('beforeunload', unsubscribe);
}

// Selectors for optimized re-renders
export const selectMessages = (state: ChatState) => state.messages;
export const selectIsLoading = (state: ChatState) => state.isLoading;
export const selectConversationId = (state: ChatState) => state.conversationId;
export const selectHasApiKey = (state: ChatState) => state.hasApiKey;
export const selectSelectedModel = (state: ChatState) => state.selectedModel;
export const selectUseAgent = (state: ChatState) => state.useAgent;
export const selectWorkflowSteps = (state: ChatState) => state.workflowSteps;
export const selectShowWorkflow = (state: ChatState) => state.showWorkflow;

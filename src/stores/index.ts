// src/stores/index.ts
// Auth Store
export { useAuthStore, selectUser, selectIsAuthenticated, selectIsAuthLoading, selectMaxFileSize } from './authStore';

// Chat Store
export {
  useChatStore,
  selectMessages,
  selectIsLoading,
  selectConversationId,
  selectHasApiKey,
  selectSelectedModel,
  selectUseAgent,
  selectWorkflowSteps,
  selectShowWorkflow,
} from './chatStore';

// Annotations Store
export { useAnnotationsStore, selectAnnotations, selectSelectedAnnotation } from './annotationsStore';

// Documents Store
export { useDocumentsStore, selectDocumentGroups, selectIsLoadingDocs } from './documentsStore';

// UI Store
export { useUIStore, selectSplitPosition, selectSidebarOpen } from './uiStore';

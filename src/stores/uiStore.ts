// src/stores/uiStore.ts
import { create } from 'zustand';
import { devtools, persist, createJSONStorage } from 'zustand/middleware';

interface UIState {
  // Layout
  splitPosition: number;
  sidebarOpen: boolean;
  pdfViewerVisible: boolean;

  // PDF Viewer
  pdfPageNumber: number;
  pdfScale: number;
  pdfRotation: number;
  showAnnotations: boolean;

  // Chat UI
  isModelMenuOpen: boolean;
  modelSearchQuery: string;

  // Actions
  setSplitPosition: (position: number) => void;
  toggleSidebar: () => void;
  setPdfViewerVisible: (visible: boolean) => void;
  togglePdfViewer: () => void;
  setPdfPage: (page: number) => void;
  setPdfScale: (scale: number) => void;
  setPdfRotation: (rotation: number) => void;
  toggleAnnotations: () => void;
  toggleModelMenu: () => void;
  setModelSearchQuery: (query: string) => void;
}

export const useUIStore = create<UIState>()(
  devtools(
    persist(
      (set) => ({
        // Initial State
        splitPosition: 60,
        sidebarOpen: true,
        pdfViewerVisible: true,
        pdfPageNumber: 1,
        pdfScale: 1.0,
        pdfRotation: 0,
        showAnnotations: true,
        isModelMenuOpen: false,
        modelSearchQuery: '',

        // Actions
        setSplitPosition: (splitPosition) => set({ splitPosition }),

        toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

        setPdfViewerVisible: (pdfViewerVisible) => set({ pdfViewerVisible }),

        togglePdfViewer: () => set((state) => ({ pdfViewerVisible: !state.pdfViewerVisible })),

        setPdfPage: (pdfPageNumber) => set({ pdfPageNumber }),

        setPdfScale: (pdfScale) => set({ pdfScale }),

        setPdfRotation: (pdfRotation) => set({ pdfRotation }),

        toggleAnnotations: () => set((state) => ({ showAnnotations: !state.showAnnotations })),

        toggleModelMenu: () => set((state) => {
          if (!state.isModelMenuOpen) {
            return { isModelMenuOpen: true, modelSearchQuery: '' };
          }
          return { isModelMenuOpen: false };
        }),

        setModelSearchQuery: (modelSearchQuery) => set({ modelSearchQuery }),
      }),
      {
        name: 'ui-storage',
        storage: createJSONStorage(() => sessionStorage),
        partialize: (state) => ({
          splitPosition: state.splitPosition,
          sidebarOpen: state.sidebarOpen,
          pdfScale: state.pdfScale,
          pdfViewerVisible: state.pdfViewerVisible,
        }),
      }
    ),
    { name: 'UIStore' }
  )
);

// Selectors
export const selectSplitPosition = (state: UIState) => state.splitPosition;
export const selectSidebarOpen = (state: UIState) => state.sidebarOpen;
export const selectPdfViewerVisible = (state: UIState) => state.pdfViewerVisible;

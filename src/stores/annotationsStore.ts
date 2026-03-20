// src/stores/annotationsStore.ts
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { AnnotationReference } from '../types';

interface HighlightRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface AnnotationsState {
  currentAnnotations: AnnotationReference[];
  highlightedTextRects: HighlightRect[];
  selectedAnnotationId: string | null;

  // Actions
  setAnnotations: (annotations: AnnotationReference[]) => void;
  addAnnotation: (annotation: AnnotationReference) => void;
  updateAnnotation: (pageNumber: number, updates: Partial<AnnotationReference>) => void;
  clearAnnotations: () => void;
  setHighlightedRects: (rects: HighlightRect[]) => void;
  setSelectedAnnotation: (id: string | null) => void;
}

export const useAnnotationsStore = create<AnnotationsState>()(
  devtools(
    (set) => ({
      currentAnnotations: [],
      highlightedTextRects: [],
      selectedAnnotationId: null,

      setAnnotations: (currentAnnotations) => set({ currentAnnotations }),

      addAnnotation: (annotation) => set((state) => ({
        currentAnnotations: [...state.currentAnnotations, annotation],
      })),

      updateAnnotation: (pageNumber, updates) => set((state) => ({
        currentAnnotations: state.currentAnnotations.map((ann) =>
          ann.pageNumber === pageNumber ? { ...ann, ...updates } : ann
        ),
      })),

      clearAnnotations: () => set({
        currentAnnotations: [],
        highlightedTextRects: [],
        selectedAnnotationId: null,
      }),

      setHighlightedRects: (highlightedTextRects) => set({ highlightedTextRects }),

      setSelectedAnnotation: (selectedAnnotationId) => set({ selectedAnnotationId }),
    }),
    { name: 'AnnotationsStore' }
  )
);

// Selectors
export const selectAnnotations = (state: AnnotationsState) => state.currentAnnotations;
export const selectSelectedAnnotation = (state: AnnotationsState) => state.selectedAnnotationId;

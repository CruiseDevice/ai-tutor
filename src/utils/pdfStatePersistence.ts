const PDF_STATE_KEY = 'pdf-viewer-state';

export interface PDFState {
  scale: number;
  rotation: number;
}

/**
 * Load the PDF viewer state from localStorage.
 * Returns null if no saved state exists.
 */
export function loadPDFState(): PDFState | null {
  if (typeof window === 'undefined') return null;

  try {
    const saved = localStorage.getItem(PDF_STATE_KEY);
    return saved ? JSON.parse(saved) : null;
  } catch {
    return null;
  }
}

/**
 * Save the PDF viewer state to localStorage.
 */
export function savePDFState(state: PDFState): void {
  if (typeof window === 'undefined') return;

  try {
    localStorage.setItem(PDF_STATE_KEY, JSON.stringify(state));
  } catch {
    // Silently fail if localStorage is not available
  }
}

/**
 * Clear the saved PDF viewer state from localStorage.
 */
export function clearPDFState(): void {
  if (typeof window === 'undefined') return;

  try {
    localStorage.removeItem(PDF_STATE_KEY);
  } catch {
    // Silently fail if localStorage is not available
  }
}

// src/config/index.ts

/**
 * Central configuration constants for the application.
 *
 * This file serves as the single source of truth for hardcoded values
 * that would otherwise be scattered across components and stores.
 */

/**
 * File upload limits in bytes
 */
export const FILE_UPLOAD_LIMITS = {
  /** Default maximum file size for regular users (10MB) */
  DEFAULT_MAX_BYTES: 10 * 1024 * 1024,

  /** Maximum file size for premium users (100MB) - future use */
  PREMIUM_MAX_BYTES: 100 * 1024 * 1024,

  /** Maximum file size for admin users (500MB) - future use */
  ADMIN_MAX_BYTES: 500 * 1024 * 1024,
} as const;

/**
 * API endpoints and timeouts
 */
export const API_CONFIG = {
  /** Default timeout for API requests in milliseconds */
  REQUEST_TIMEOUT_MS: 30000,

  /** SSE (Server-Sent Events) reconnection delay in milliseconds */
  SSE_RECONNECT_DELAY_MS: 1000,
} as const;

/**
 * UI defaults
 */
export const UI_DEFAULTS = {
  /** Default split position for resizable panes (percentage) */
  SPLIT_POSITION: 60,

  /** Default PDF viewer scale */
  PDF_SCALE: 1.0,

  /** Default PDF rotation in degrees */
  PDF_ROTATION: 0,
} as const;

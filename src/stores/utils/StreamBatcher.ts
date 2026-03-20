// src/stores/utils/StreamBatcher.ts

/**
 * Batches streaming updates to reduce React re-renders during SSE streaming.
 *
 * Without batching: Each SSE chunk triggers a React update (50-100 updates/sec).
 * With batching: Updates are flushed every RAF frame (~16ms) for ~60 FPS.
 *
 * Usage:
 *   const batcher = new StreamBatcher((id, content) => {
 *     set((state) => ({
 *       messages: state.messages.map(msg =>
 *         msg.id === id ? { ...msg, content } : msg
 *       ),
 *     }));
 *   });
 *
 *   // During streaming
 *   batcher.addChunk(messageId, chunk);
 *
 *   // On stream complete
 *   batcher.complete(messageId);
 */
export class StreamBatcher {
  private buffer = new Map<string, string>();
  private rafId: number | null = null;
  private onUpdate: (id: string, content: string) => void;

  /**
   * @param onUpdate - Callback fired with (messageId, accumulatedContent) on each flush
   * @param flushInterval - Optional interval in ms (default: RAF ~16ms)
   */
  constructor(
    onUpdate: (id: string, content: string) => void,
    private flushInterval: number = -1 // -1 means use RAF
  ) {
    this.onUpdate = onUpdate;
  }

  /**
   * Add a chunk to the buffer for a specific message.
   * Chunks are accumulated and flushed on the next RAF frame.
   */
  addChunk(messageId: string, chunk: string): void {
    const current = this.buffer.get(messageId) || '';
    this.buffer.set(messageId, current + chunk);

    if (!this.rafId) {
      this.scheduleFlush();
    }
  }

  /**
   * Schedule a flush using RAF or setTimeout.
   */
  private scheduleFlush(): void {
    if (this.flushInterval < 0) {
      // Use requestAnimationFrame for browser paint cycle sync
      this.rafId = requestAnimationFrame(() => {
        this.flush();
        this.rafId = null;
      });
    } else {
      // Use setTimeout for custom interval (mainly for testing)
      this.rafId = window.setTimeout(() => {
        this.flush();
        this.rafId = null;
      }, this.flushInterval) as unknown as number;
    }
  }

  /**
   * Flush all buffered updates to the store.
   */
  flush(): void {
    Array.from(this.buffer.entries()).forEach(([id, content]) => {
      this.onUpdate(id, content);
    });
    this.buffer.clear();
  }

  /**
   * Mark a message as complete.
   * Performs a final flush and clears the message from buffer.
   */
  complete(messageId: string): void {
    this.flush();
    this.buffer.delete(messageId);
  }

  /**
   * Cancel any pending flush and clear buffer.
   * Call this on stream error to prevent orphaned updates.
   */
  destroy(): void {
    if (this.rafId !== null) {
      if (this.flushInterval < 0) {
        cancelAnimationFrame(this.rafId);
      } else {
        clearTimeout(this.rafId);
      }
      this.rafId = null;
    }
    this.buffer.clear();
  }
}

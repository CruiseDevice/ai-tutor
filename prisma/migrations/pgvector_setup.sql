-- Enable pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Make sure search path is set correctly
SELECT set_config('search_path', 'public,extensions', false);

-- Ensure indexes are present for vector search
CREATE INDEX IF NOT EXISTS document_chunk_embedding_idx ON "DocumentChunk" USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
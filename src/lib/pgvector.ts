// lib/pgvector.ts
import { OpenAIEmbeddings } from "@langchain/openai";
import prisma from "./db";
import crypto from "crypto";

interface ChunkData {
  content: string;
  pageNumber: number;
  embedding: number[];
  documentId: string;
}

// save document chunks with embeddings to PostgreSQL
export async function saveDocumentChunks(chunks: ChunkData[]) {
  try {
    // Use raw SQL to insert chunks with vector data
    for (const chunk of chunks) {
      await prisma.$executeRaw`
        INSERT INTO "DocumentChunk" (id, content, "pageNumber", embedding, "documentId", "createdAt", "updatedAt")
        VALUES (
          ${crypto.randomUUID()}, 
          ${chunk.content}, 
          ${chunk.pageNumber}, 
          ${chunk.embedding}::vector, 
          ${chunk.documentId}, 
          NOW(), 
          NOW()
        )
      `
    }
    return {success: true, count: chunks.length};
  } catch (error) {
    console.error("Error saving document chunks: ", error);
    throw error;
  }
}

export async function findSimilarChunks (
  query: string,
  documentId: string,
  limit: number=5,
  apiKey: string
) {
  try {
    if (!query || !documentId || !apiKey) {
      throw new Error("Missing required parameters for vector search");
    }
    // create embeddings for the query
    const embeddings = new OpenAIEmbeddings({
        openAIApiKey: apiKey,
        modelName: "text-embedding-3-small"
    });

    const queryEmbedding = await embeddings.embedQuery(query);

    if (!queryEmbedding || queryEmbedding.length === 0) {
      throw new Error("Failed to generate embeddings for the query");
    }

    // Use raw SQL to perform vector similarity search with proper casting
    const chunks = await prisma.$queryRaw`
    SELECT 
      id, 
      content, 
      "pageNumber"::integer as "pageNumber",
      "documentId",
      1 - (embedding::vector <=> ${queryEmbedding}::vector) as similarity
    FROM "DocumentChunk"
    WHERE "documentId" = ${documentId}
    ORDER BY similarity DESC
    LIMIT ${limit};
    `;
    
    // Debug logging to check the returned data structure
    if (Array.isArray(chunks) && chunks.length > 0) {
      console.log("Sample chunk from pgvector:", {
        pageNumberValue: chunks[0].pageNumber,
        pageNumberType: typeof chunks[0].pageNumber,
        hasPageNumber: 'pageNumber' in chunks[0]
      });
    }

    return Array.isArray(chunks) ? chunks : [];
  } catch (error) {
    console.error("Error finding similar chunks:", error instanceof Error ? error.message : String(error));
    return [];
  }
}
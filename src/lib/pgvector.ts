// lib/pgvector.ts
import prisma from "./db";
import crypto from "crypto";

const EMBEDDINGS_SERVICE_URL = process.env.EMBEDDINGS_SERVICE_URL || "http://localhost:8000";

// Interface to include position data
interface ChunkData {
  content: string;
  pageNumber: number;
  embedding: number[];
  documentId: string;
  // Store text ranges for annotations
  textRanges?: {
    from: number;
    to: number;
  }[];
}

// function to get embeddings from our service
async function getEmbedding(text: string): Promise<number[]> {
  try {
    const response = await fetch(`${EMBEDDINGS_SERVICE_URL}/embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({text})
    });

    if (!response.ok) {
      throw new Error(`Failed to get embedding: ${response.statusText}`);
    }

    const data = await response.json();
    return data.embedding;
    
  } catch (error) {
    console.error("Error getting embedding: ", error);
    throw error;
  }
}

// Function to get batch embeddings from the embeddings service
export async function getBatchEmbeddings(texts: string[]): Promise<number[][]> {
  try{
    const response = await fetch(`${EMBEDDINGS_SERVICE_URL}/batch-embeddings`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({texts})
    });

    if (!response.ok) {
      throw new Error("Failed to get batch embeddings: ${response.statusText}");
    }

    const data = await response.json();
    return data.embeddings;
  } catch (error) {
    console.error("Error getting batch embeddings: ", error);
    throw error;
  }
}

// save document chunks with embeddings to PostgreSQL
export async function saveDocumentChunks(chunks: ChunkData[]) {
  try {
    // Use raw SQL to insert chunks with vector data
    for (const chunk of chunks) {
      // Convert textRanges to JSON if it exists
      const positionData = chunk.textRanges ? {textRanges: chunk.textRanges} : null;

      await prisma.$executeRaw`
        INSERT INTO "DocumentChunk" (id, content, "pageNumber", embedding, "documentId", "positionData", "createdAt", "updatedAt")
        VALUES (
          ${crypto.randomUUID()}, 
          ${chunk.content}, 
          ${chunk.pageNumber}, 
          ${chunk.embedding}::vector,
          ${chunk.documentId},
          ${positionData ? JSON.stringify(positionData) : null}::jsonb,
          NOW(),
          NOW()
        )
      `;
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

    const queryEmbedding = await getEmbedding(query);

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
      "positionData",
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
        hasPageNumber: 'pageNumber' in chunks[0],
        hasPositionData: 'positionData' in chunks[0],
        positionSample: chunks[0].positionData
      });
    }

    return Array.isArray(chunks) ? chunks : [];
  } catch (error) {
    console.error("Error finding similar chunks:", error instanceof Error ? error.message : String(error));
    return [];
  }
}


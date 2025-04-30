// app/api/documents/process/route.ts

import prisma from "@/lib/db";
import { NextRequest, NextResponse } from "next/server";
import { getSignedS3Url } from "@/lib/s3";
import crypto from "crypto";

// Embeddings service URL
const EMBEDDINGS_SERVICE_URL = process.env.EMBEDDINGS_SERVICE_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const { documentId } = await req.json();
    console.log('Starting document processing for documentId: ', documentId);

    // get document and user info
    const document = await prisma.document.findUnique({
      where: { id: documentId },
      include: { user: true }
    });

    if (!document) {
      return NextResponse.json(
        { error: 'Document not found' },
        { status: 404 }
      );
    }

    // Generate a fresh signed URL for the PDF
    let pdfUrl = document.url;
    if (!pdfUrl.includes('X-Amz-Signature=')) {
      pdfUrl = await getSignedS3Url(document.blobPath);
    }

    // Fetch the PDF
    const pdfResponse = await fetch(pdfUrl);
    if (!pdfResponse.ok) {
      throw new Error(`Failed to fetch PDF: ${pdfResponse.statusText}`);
    }
    
    // Get the PDF as a blob
    const pdfBlob = await pdfResponse.blob();
    
    // Create form data to send to the FastAPI service
    const formData = new FormData();
    formData.append('file', pdfBlob, 'document.pdf');
    formData.append('document_id', document.id);
    
    // Send the PDF to the FastAPI service for processing
    console.log('Sending PDF to FastAPI service for processing...');
    
    try {

      const processingResponse = await fetch(`${EMBEDDINGS_SERVICE_URL}/process-document`, {
        method: 'POST',
        body: formData,
      });
      
      if (!processingResponse.ok) {
        const errorDetail = await processingResponse.text();
        throw new Error(`FastAPI service error: ${errorDetail}`);
      }
      
      // Get the processing results
      const processingResult = await processingResponse.json();
      console.log('Processing result:', processingResult);
      
      // Save document chunks to database using Prisma
      if (processingResult.chunks && processingResult.chunks.length > 0) {
        console.log(`Saving ${processingResult.chunks.length} chunks to database...`);
        
        // Save each chunk to the database
        for (const chunk of processingResult.chunks) {
          // Prepare position data for PostgreSQL
          const positionData = chunk.text_ranges ? { textRanges: chunk.text_ranges } : null;
          
          // We need to use raw SQL for the vector type insertion
          // Prisma does not directly handle the vector type through its API
          await prisma.$executeRaw`
            INSERT INTO "DocumentChunk" (
              id, 
              content, 
              "pageNumber", 
              embedding, 
              "documentId", 
              "positionData", 
              "createdAt", 
              "updatedAt"
            )
            VALUES (
              ${String(crypto.randomUUID())}, 
              ${chunk.content}, 
              ${chunk.page_number}, 
              ${chunk.embedding}::vector(768),
              ${documentId},
              ${positionData ? JSON.stringify(positionData) : null}::jsonb,
              NOW(),
              NOW()
            )
          `;
        }
      }

      console.log(`Document processing complete. Successfully processed ${processingResult.chunks_processed} chunks. Failed: ${processingResult.chunks_failed}`);
      return NextResponse.json({
        success: true,
        message: 'Document processed successfully',
        chunksProcessed: processingResult.chunks_processed,
        chunksFailed: processingResult.chunks_failed
      });

    } catch (error) {
      throw error;
    }

  } catch (error) {
    console.error('Error processing document: ', error);
    return NextResponse.json(
      { error: 'Failed to process document' },
      { status: 500 }
    )
  }
}
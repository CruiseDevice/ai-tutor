// app/api/documents/process/route.ts

import prisma from "@/lib/db";
import { NextRequest, NextResponse } from "next/server";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { saveDocumentChunks } from "@/lib/pgvector";

export async function POST(req: NextRequest) {
  try {
    const { documentId } = await req.json();
    console.log('Starting document processing for documentId: ', documentId);

    // get document and user info
    const document = await prisma.document.findUnique({
      where: {id: documentId},
      include: {user: true}
    });

    if (!document) {
      return NextResponse.json(
        {error: 'Document not found'},
        {status: 404}
      );
    }

    if (!document?.user?.apiKey){
      console.log('OpenAI API key not found for user');
      return NextResponse.json(
        {error: 'OpenAI API key not found'},
        {status: 400}
      );
    }

    // Initialize OpenAI embeddings
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: document.user.apiKey,
      modelName: "text-embedding-3-small"
    });

    // Load pdf
    const response = await fetch(document.url);
    if(!response.ok) {
      throw new Error(`Failed to fetch PDF: ${response.statusText}`);
    }
    const pdfBlob = await response.blob();
    const loader = new PDFLoader(pdfBlob, {
      splitPages: true,
    });

    console.log("Starting PDF processing...");

    // load and process the pdf
    const rawDocs = await loader.load();
    console.log('PDF loaded, number of pages: ', rawDocs.length);

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    })

    const docs = await textSplitter.splitDocuments(rawDocs);
    console.log('Split into chunks: ', docs.length);

    // process chunks and store them with embeddings
    console.log('Starting to process and store chunks...');
    let successfulChunks = 0;
    let failedChunks = 0;

    // prepare chunks with embeddings for pgvector
    const chunksWithEmbeddings = [];

    for(let i = 0; i < docs.length; i++) {
      try {
        const doc = docs[i];
        const vectorEmbedding = await embeddings.embedQuery(doc.pageContent);
        
        let pageNumber = 1;

        if(doc.metadata && doc.metadata.loc.pageNumber !== undefined) {
          // Try to parse the page number as an integer
          pageNumber = parseInt(String(doc.metadata.loc.pageNumber), 10);
        }

        // Extract line positions from metadata if available
        const textRanges = [];
        if(doc.metadata && doc.metadata.loc && doc.metadata.loc.lines) {
          const lines = doc.metadata.loc.lines;
          // check if lines is an object with from/to properties
          if(lines.from !== undefined && lines.to !== undefined) {
            textRanges.push({
              from: parseInt(String(lines.from), 10),
              to: parseInt(String(lines.to), 10)
            });
          }
          console.log(`Extracted text ranges from chunk on page: ${pageNumber}: `, textRanges);
        }
        chunksWithEmbeddings.push({
          content: doc.pageContent,
          pageNumber: pageNumber,
          embedding: vectorEmbedding,
          documentId,
          textRanges: textRanges.length > 0 ? textRanges : undefined
        });

        successfulChunks++;
      } catch (error) {
        console.error('Error processing chunks: ', error);
        failedChunks++;
      }
    }
    
    // save chunks to PostgreSQL with pgvector
    if (chunksWithEmbeddings.length > 0) {
      await saveDocumentChunks(chunksWithEmbeddings);
    }

    console.log(`Document processing complete. Successfully processed ${successfulChunks} chunks. Failed: ${failedChunks}`);
    return NextResponse.json({
      success: true,
      message: 'Document processed successfully',
      chunksProcessed: successfulChunks,
      chunksFailed: failedChunks
    });

  } catch (error) {
    console.error('Error processing document: ', error);
    return NextResponse.json(
      {error: 'Failed to process document'},
      {status: 500}
    )
  }
}
import prisma from "@/lib/db";
import { NextRequest, NextResponse } from "next/server";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

export async function POST(req: NextRequest) {
  try {

     // Get session token from cookie
     const sessionToken = req.headers.get('cookie')
     ?.split(';')
     ?.find(c => c.trim().startsWith('session_token='))
     ?.split('=')[1];

    if (!sessionToken) {
      return NextResponse.json({ error: 'No session token' }, { status: 401 });
    }

    // Verify session
    const session = await prisma.session.findFirst({
      where: {
        token: sessionToken,
        expiresAt: {
          gt: new Date(),
        },
      },
      include: {
        user: true,
      },
    });

    if (!session) {
      return NextResponse.json({ error: 'Invalid session' }, { status: 401 });
    }
    
    const { documentId } = await req.json();
    console.log('Starting document processing for documentId: ', documentId);

    // get document and user info
    const document = await prisma.document.findUnique({
      where: {id: documentId},
      include: {user: true}
    });

    console.log('Found document: ', document?.title);

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
    console.log('Fetching PDF from URL: ', document.url);
    const response = await fetch(document.url);
    if(!response.ok) {
      throw new Error(`Failed to fetch PDF: ${response.statusText}`);
    }
    const pdfBlob = await response.blob();
    const loader = new PDFLoader(pdfBlob);

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

    for(const doc of docs) {
      try{
        const vectorEmbedding = await embeddings.embedQuery(doc.pageContent);
        await prisma.documentChunk.create({
          data: {
            content: doc.pageContent,
            pageNumber: doc.metadata.pageNumber || 1,
            embedding: vectorEmbedding,
            documentId: documentId
          }
        });
        successfulChunks++;
      } catch (error) {
        console.error('Error processing chunk: ', error);
        failedChunks++;
      }
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
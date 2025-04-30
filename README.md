# StudyFetch AI Tutor

StudyFetch AI Tutor is a web application that helps students understand PDF documents through an interactive split-screen interface. Users can upload PDFs and chat with an AI about the document's content, with the AI able to reference and highlight relevant parts of the PDF in real-time.

## Features

- 🔐 **User Authentication**: Secure email/password signup and login with session management
- 📄 **PDF Upload & Viewing**: Upload, store, and navigate PDF documents
- 💬 **AI Chat Interface**: Interact with the AI about document content via text
- 🔍 **Smart Document Search**: Vector embeddings power semantic retrieval of relevant document content
- 📌 **Context-Aware Responses**: AI references specific page numbers and content from the PDF
- 📝 **Persistent Conversations**: Chat history is saved and can be resumed later
- 🔄 **Multi-Document Support**: Upload and manage multiple documents with separate conversation histories

## Tech Stack

### Frontend
- **Next.js 15+** with App Router
- **React 19**
- **TailwindCSS** for styling
- **React PDF** for PDF rendering

### Backend
- **Next.js API Routes**
- **PostgreSQL** with pgvector extension for vector similarity search
- **Prisma ORM** for database operations
- **AWS S3** for PDF storage

### AI Integration
- **OpenAI GPT-4** for chat responses
- **Custom Embeddings Service** using sentence-transformers
- **LangChain** for document processing

## Architecture

### RAG Pipeline
![RAG Pipeline](./rag-diagram-1.webp)
*Image credit: https://www.dailydoseofds.com/*

The application follows a Retrieval Augmented Generation (RAG) approach:
1. PDF documents are processed into chunks
2. Each chunk gets a vector embedding representing its semantic meaning
3. When the user asks a question, relevant chunks are retrieved via similarity search
4. The AI generates a response based on the retrieved context

### Service Components
- **Web Application**: Next.js app for frontend and API routes
- **Embeddings Service**: FastAPI service for document processing and embedding generation
- **PostgreSQL Database**: Stores user data, documents, conversations, and vector embeddings

## Prerequisites

- Node.js v18+
- Docker and Docker Compose
- OpenAI API key
- AWS S3 credentials (for production deployment)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CruiseDevice/ai-tutor
cd ai-tutor
```

### 2. Install dependencies

```bash
npm install
```

### 3. Start the database and embeddings service

```bash
docker-compose up -d
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
# Database
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/studyfetch"

# S3 Storage
AWS_REGION="us-east-1"
AWS_ACCESS_KEY_ID="your-access-key"
AWS_SECRET_ACCESS_KEY="your-secret-key"
S3_PDFBUCKET_NAME="your-bucket-name"

# Embeddings Service
EMBEDDINGS_SERVICE_URL="http://localhost:8000"

# Environment
NODE_ENV="development"
```

### 5. Initialize the database

```bash
npx prisma generate
npx prisma db push
node scripts/setup-pgvector.js
```

## Running the Application

Start the development server:

```bash
npm run dev
```

The application will be available at [http://localhost:3000](http://localhost:3000)

## User Guide

1. **Register/Login**: Create an account or sign in
2. **API Setup**: Navigate to API Settings and add your OpenAI API key
3. **Upload a PDF**: On the dashboard, click "Upload PDF" to begin
4. **Chat with the Document**: Ask questions about the PDF content
5. **Document History**: Access previous documents from the sidebar

## Project Structure

```
/
├── embeddings/                # Embeddings service (FastAPI)
│   ├── document_processor.py  # PDF processing logic
│   └── embeddings_service.py  # API endpoints
├── prisma/                    # Database schema and migrations
├── public/                    # Static assets
├── scripts/                   # Setup scripts
├── src/
│   ├── app/                   # Next.js app directory
│   │   ├── api/               # API routes
│   │   ├── dashboard/         # Dashboard page
│   │   ├── login/             # Login page
│   │   └── register/          # Registration page
│   ├── components/            # React components
│   │   ├── ChatInterface.tsx  # Chat UI
│   │   ├── Dashboard.tsx      # Main application component
│   │   └── EnhancedPDFViewer.tsx # PDF viewer with annotation
│   └── lib/                   # Utility libraries
│       ├── auth.ts            # Authentication utilities
│       ├── db.ts              # Database client
│       └── pgvector.ts        # Vector search functions
└── docker-compose.yml         # Docker services configuration
```

## API Routes

- `/api/auth/*`: Authentication endpoints (login, register, logout)
- `/api/documents`: PDF upload and processing
- `/api/conversations`: Conversation management
- `/api/chat`: AI messaging endpoint

## Development

### Add Database Migration

After schema changes:

```bash
npx prisma migrate dev --name your_migration_name
```

### Update Embeddings Model

To change the embeddings model, update the model name in:
- `/embeddings/document_processor.py`
- `/embeddings/embeddings_service.py`

## Deployment

The application can be deployed on Vercel with the following considerations:

1. Set up a PostgreSQL database with pgvector extension (e.g., using Supabase or Neon)
2. Deploy the embeddings service separately (e.g., on a server or containerized service)
3. Configure environment variables in your hosting platform

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- OpenAI for the GPT API
- Sentence Transformers for embeddings
- LangChain for document processing utilities
- Vercel for Next.js hosting infrastructure
# StudyFetch AI Tutor

StudyFetch AI Tutor is a web application that helps students understand PDF documents through an interactive split-screen interface. Users can upload PDFs and chat with an AI about the document's content, with the AI able to reference and highlight relevant parts of the PDF in real-time.

![Landing Page](./landing_page.png)

## Features

- 🔐 **User Authentication**: Secure email/password signup and login with JWT-based session management, password reset flow
- 📄 **PDF Upload & Viewing**: Upload, store, and navigate PDF documents with streaming processing
- 💬 **AI Chat Interface**: Interact with the AI about document content via text with streaming responses
- 🔍 **Advanced Document Search**:
  - **Hybrid Search**: Combines semantic (vector) and keyword (full-text) search for better retrieval
  - **Re-ranking**: Cross-encoder re-ranking improves result quality
  - **Query Expansion**: Optional multi-query retrieval for complex queries
- 📌 **Context-Aware Responses**: AI references specific page numbers and content from the PDF
- 📝 **Persistent Conversations**: Chat history is saved and can be resumed later
- 🔄 **Multi-Document Support**: Upload and manage multiple documents with separate conversation histories
- ⚡ **Performance Optimizations**: Redis caching, background workers, rate limiting
- 👁️ **Vision Support**: Multi-modal document understanding with image extraction and analysis
- 🔧 **Admin Panel**: Administrative interface for system management

## Tech Stack

### Frontend
- **Next.js 15+** with App Router
- **React 19**
- **TailwindCSS** for styling
- **React PDF** for PDF rendering
- **React Markdown** for formatted AI responses

### Backend
- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - ORM for database operations
- **PostgreSQL** with pgvector extension for vector similarity search
- **Redis** - Caching and rate limiting
- **ARQ** - Background job processing for document uploads
- **AWS S3** - PDF file storage
- **JWT** - Token-based authentication

The backend is split into two Python services:
- **Main Backend** (`backend`, port 8001) – FastAPI app with business logic and database access (no heavy ML dependencies)
- **Embedding Service** (`embedding-service`, port 8002) – FastAPI microservice that hosts PyTorch/sentence-transformers models for embeddings and reranking

### AI Integration
- **OpenAI GPT-4/GPT-4o-mini** - Chat completions with streaming support
- **sentence-transformers** - Embedding generation (all-mpnet-base-v2, 768 dimensions) running inside the embedding service
- **LangChain** - Document processing and core RAG utilities
- **Cross-Encoder Re-ranking** - ms-marco-MiniLM-L-6-v2 for improved retrieval quality
- **LangGraph-based Agent Workflow** - Optional multi-step RAG agent pipeline

## Architecture

### RAG Pipeline
![RAG Pipeline](./rag-diagram-1.webp)
*Image credit: https://www.dailydoseofds.com/*

The application follows an advanced Retrieval Augmented Generation (RAG) approach:

1. **Document Processing**: PDF documents are processed into chunks with streaming support
2. **Embedding Generation**: Each chunk gets a vector embedding representing its semantic meaning
3. **Hybrid Retrieval**: When the user asks a question:
   - Query embedding is generated
   - **Semantic Search**: Vector similarity search using pgvector
   - **Keyword Search**: PostgreSQL full-text search (optional)
   - **Hybrid Fusion**: Combines both results with weighted scoring
   - **Re-ranking**: Cross-encoder re-ranks top candidates for better relevance
4. **Response Generation**: The AI generates a streaming response based on retrieved context with citations

### Service Components
- **Frontend**: Next.js app (UI only, no API routes)
- **Main Backend API**: FastAPI server on port 8001 (business logic, authentication, RAG orchestration)
- **Embedding Service**: Separate FastAPI microservice on port 8002 for embeddings and reranking
- **Background Workers**: ARQ workers for async document processing
- **PostgreSQL Database**: Stores user data, documents, conversations, and vector embeddings
- **Redis**: Caching layer and rate limiting

The LangGraph-based agent workflow (via `RAGAgentService`) runs inside the main backend and can be toggled per request using the `use_agent` flag in chat requests.

## Prerequisites

- **Node.js** v18+
- **Python** 3.9+
- **Docker** and Docker Compose
- **OpenAI API key**
- **AWS S3 credentials** (for production deployment)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CruiseDevice/ai-tutor
cd ai-tutor
```

### 2. Install frontend dependencies

```bash
npm install
```

### 3. (Optional) Install backend dependencies for local (non-Docker) development

```bash
cd backend
pip install -r requirements.txt
cd ..
```

### 4. Start backend services with Docker Compose

```bash
# First-time build (builds backend + embedding service)
docker-compose down -v  # optional cleanup
DOCKER_BUILDKIT=1 docker-compose up --build

# Subsequent runs (reuses cached embedding service image)
docker-compose up -d

# Verify services are running
curl http://localhost:8001/health  # Backend API
curl http://localhost:5432         # PostgreSQL (should fail, but confirms port is open)
```

The Docker Compose setup includes:
- **PostgreSQL** (`db`, port 5432) - Database with pgvector extension
- **Redis** (`redis`, port 6379) - Caching and rate limiting
- **Embedding Service** (`embedding-service`, port 8002) - ML microservice for embeddings and reranking
- **Main Backend API** (`backend`, port 8001) - FastAPI server
- **Worker** (`worker`) - ARQ background worker for document processing

### 5. Set up environment variables

Create a `.env` file in the root directory:

```env
# Database
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/studyfetch"

# Redis
REDIS_URL="redis://localhost:6379/0"

# S3 Storage
AWS_REGION="us-east-1"
AWS_ACCESS_KEY_ID="your-access-key"
AWS_SECRET_ACCESS_KEY="your-secret-key"
S3_PDFBUCKET_NAME="your-bucket-name"

# Security
JWT_SECRET="change-this-secret-in-production"
ENCRYPTION_KEY="your-32-byte-encryption-key-for-api-keys"

# Backend URL (for frontend)
NEXT_PUBLIC_BACKEND_URL="http://localhost:8001"

# Environment
NODE_ENV="development"
```

Docker Compose will automatically pick up `JWT_SECRET`, `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_PDFBUCKET_NAME`, and `NODE_ENV` from this `.env` file when bringing up the backend and worker containers.

### 6. Initialize the database

The database tables are automatically created on backend startup. The backend includes automatic migrations for:
- pgvector extension setup
- HNSW indexes for vector search
- Full-text search indexes for hybrid search
- Document status fields
- Other schema updates

If you need to manually verify:

```bash
# Connect to PostgreSQL
docker exec -it study-fetch-tutor-db-1 psql -U postgres -d studyfetch

# Check pgvector extension
\dx

# Check tables
\dt
```

## Running the Application

### Development Mode (recommended: Docker for backend + local frontend)

1. **Start backend stack with Docker** (from repo root):
   ```bash
   DOCKER_BUILDKIT=1 docker-compose up --build
   ```
   This starts PostgreSQL, Redis, embedding service, backend API, and worker.

2. **Start the frontend locally**:
   ```bash
   npm run dev
   ```

3. Visit the app at:
   - **Frontend**: [http://localhost:3000](http://localhost:3000)
   - **Backend API** (in Docker): [http://localhost:8001](http://localhost:8001)
   - **Embedding Service** (in Docker): [http://localhost:8002/health](http://localhost:8002/health)

### Development Mode (backend without Docker)

1. **Start the backend directly**:
   ```bash
   cd backend
   uvicorn app.main:app --reload --port 8001
   ```

2. **Start the frontend**:
   ```bash
   npm run dev
   ```

3. **Start the worker** (if not using Docker):
   ```bash
   cd backend
   arq app.workers.arq_config.WorkerSettings
   ```

The application will be available at:
- **Frontend**: [http://localhost:3000](http://localhost:3000)
- **Backend API**: [http://localhost:8001](http://localhost:8001)
- **API Documentation**: [http://localhost:8001/docs](http://localhost:8001/docs) (Swagger UI)

### Production Mode

Use Docker Compose for production deployment:

```bash
docker-compose up -d
```

Make sure to set appropriate environment variables in your `.env` file, especially:
- `JWT_SECRET` - Use a strong random secret
- `ENCRYPTION_KEY` - 32-byte key for API key encryption
- `COOKIE_SECURE=True` - For HTTPS
- `NODE_ENV=production`

## User Guide

1. **Register/Login**: Create an account or sign in
2. **API Setup**: Navigate to Settings and add your OpenAI API key (encrypted and stored securely)
3. **Upload a PDF**: On the dashboard, click "Upload PDF" to begin
4. **Wait for Processing**: Documents are processed in the background (check status in the UI)
5. **Chat with the Document**: Ask questions about the PDF content
6. **Document History**: Access previous documents and conversations from the sidebar

## Project Structure

```
/
├── backend/                      # Python FastAPI Backend
│   ├── app/
│   │   ├── api/                 # API route handlers
│   │   │   ├── auth.py          # Authentication endpoints
│   │   │   ├── chat.py          # Chat endpoints (with streaming)
│   │   │   ├── documents.py     # Document upload/management
│   │   │   ├── conversations.py # Conversation management
│   │   │   └── user.py          # User profile/settings
│   │   ├── core/                # Core utilities
│   │   │   ├── security.py      # JWT, password hashing
│   │   │   ├── deps.py          # FastAPI dependencies
│   │   │   └── rate_limiting.py # Rate limiting middleware
│   │   ├── models/              # SQLAlchemy models
│   │   │   ├── user.py
│   │   │   ├── document.py
│   │   │   └── conversation.py
│   │   ├── schemas/             # Pydantic schemas
│   │   ├── services/            # Business logic
│   │   │   ├── chat_service.py  # RAG pipeline with hybrid search
│   │   │   ├── document_service.py # PDF processing
│   │   │   ├── embedding_service.py # Embedding generation
│   │   │   ├── rerank_service.py # Cross-encoder re-ranking
│   │   │   ├── cache_service.py # Redis caching
│   │   │   ├── auth_service.py  # Authentication logic
│   │   │   ├── agent_service.py # LangGraph-based RAG agent workflow
│   │   │   ├── audit_service.py # Audit logging
│   │   │   ├── contextual_compression_service.py # Context compression for RAG
│   │   │   ├── density_calculator_service.py # Query density calculator
│   │   │   ├── docling_service.py # Document parsing with Docling
│   │   │   ├── hierarchical_chunking_service.py # Hierarchical chunking
│   │   │   ├── query_decomposition_service.py # Query decomposition
│   │   │   ├── query_expansion_service.py # Query expansion for retrieval
│   │   │   ├── sentence_retrieval_service.py # Sentence-level retrieval
│   │   │   ├── vision_service.py # Vision/multimodal capabilities
│   │   │   ├── token_service.py # Token counting utilities
│   │   │   ├── encryption_service.py # API key encryption
│   │   │   └── queue_service.py # Background queue management
│   │   ├── workers/             # Background jobs
│   │   │   ├── arq_config.py    # ARQ configuration
│   │   │   └── document_jobs.py # Document processing jobs
│   │   ├── database.py          # Database connection
│   │   ├── database_migrations.py # Schema migrations
│   │   ├── config.py            # Settings and configuration
│   │   └── main.py              # FastAPI app entry point
│   ├── Dockerfile
│   └── requirements.txt
│
├── src/                          # Next.js Frontend
│   ├── app/                     # Next.js app directory
│   │   ├── dashboard/           # Dashboard page
│   │   ├── login/               # Login page
│   │   ├── register/            # Registration page
│   │   ├── settings/            # Settings page
│   │   ├── forgot-password/     # Password reset request page
│   │   ├── reset-password/      # Password reset confirmation page
│   │   ├── admin/               # Admin panel
│   │   └── layout.tsx           # Root layout
│   ├── components/              # React components
│   │   ├── ChatInterface.tsx    # Chat UI with streaming
│   │   ├── Dashboard.tsx        # Main application component
│   │   ├── EnhancedPDFViewer.tsx # PDF viewer with annotations
│   │   ├── LoginForm.tsx        # Login form
│   │   ├── RegisterForm.tsx     # Registration form
│   │   ├── ForgotPasswordForm.tsx # Password reset request form
│   │   ├── ResetPasswordForm.tsx  # Password reset confirmation form
│   │   ├── APISettings.tsx      # API key settings component
│   │   ├── AgentWorkflowProgress.tsx # Agent workflow progress indicator
│   │   ├── ChatSidebar.tsx      # Conversation history sidebar
│   │   ├── admin/               # Admin components
│   │   └── ...
│   └── lib/
│       └── api-client.ts        # Backend API client
│
├── docker-compose.yml            # Docker services configuration
└── package.json                  # Frontend dependencies
```

## API Endpoints

All backend APIs are available at `http://localhost:8001`:

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login (returns JWT cookie)
- `POST /api/auth/logout` - User logout
- `GET /api/auth/me` - Get current user
- `GET /api/auth/verify-session` - Verify session
- `POST /api/auth/password-reset/request` - Request password reset
- `POST /api/auth/password-reset/confirm` - Confirm password reset

### Documents
- `POST /api/documents` - Upload PDF
- `POST /api/documents/process` - Process document (triggers background job)
- `GET /api/documents` - List user's documents
- `GET /api/documents/{id}` - Get specific document with signed S3 URL
- `DELETE /api/documents/{id}` - Delete document

### Chat & Conversations
- `POST /api/chat` - Send message and get AI response
- `POST /api/chat/stream` - Send message and get streaming AI response (SSE)
- `GET /api/conversations` - List conversations
- `GET /api/conversations/{id}` - Get conversation with messages
- `DELETE /api/conversations/{id}` - Delete conversation

### User Settings
- `GET /api/user/profile` - Get user profile
- `PUT /api/user/profile` - Update profile
- `POST /api/user/apikey` - Update OpenAI API key (encrypted)
- `GET /api/user/apikey/check` - Check if API key exists
- `DELETE /api/user/apikey` - Delete API key

### Configuration
- `GET /api/config` - Get backend configuration

## Advanced Features

### Hybrid Search
Combines semantic (vector) and keyword (full-text) search for improved retrieval:
- Configurable weights: `SEMANTIC_SEARCH_WEIGHT` (default: 0.7) and `KEYWORD_SEARCH_WEIGHT` (default: 0.3)
- Falls back to semantic-only if keyword search fails

### Re-ranking
Cross-encoder re-ranking improves retrieval quality:
- Enabled by default: `RERANK_ENABLED=True`
- Retrieves top 20 candidates, re-ranks, returns top 5
- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` model

### Query Expansion
Optional multi-query retrieval for complex queries:
- Disabled by default: `QUERY_EXPANSION_ENABLED=False`
- Generates query variations using GPT-4o-mini
- Uses Reciprocal Rank Fusion (RRF) to combine results

### Caching
Redis-based caching for performance:
- **Embedding Cache**: Caches query embeddings (24h TTL)
- **Response Cache**: Caches similar queries (2h TTL)
- **Chunk Cache**: Caches retrieval results (72h TTL)
- Compression for large cache values

### Streaming
- **Document Processing**: Streaming PDF processing for progressive availability
- **Chat Responses**: Server-Sent Events (SSE) for real-time streaming responses

## Development

### Running Tests

```bash
cd backend
python -m pytest test_*.py
```

### Database Migrations

Migrations are handled automatically on startup via `database_migrations.py`. For manual migrations:

```python
# In Python shell or script
from app.database import engine
from app.database_migrations import add_pgvector_hnsw_index
add_pgvector_hnsw_index(engine)
```

### Update Embeddings Model

To change the embeddings model, update in `backend/app/services/embedding_service.py`:

```python
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
```

### Update Re-ranking Model

To change the re-ranking model, update in `backend/app/config.py`:

```python
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

### Environment Configuration

See `backend/app/config.py` for all available configuration options:
- Rate limiting settings
- Cache TTLs and thresholds
- Search weights and parameters
- Feature flags (re-ranking, query expansion, streaming)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- OpenAI for the GPT API
- Sentence Transformers for embeddings
- LangChain for document processing utilities
- FastAPI for the excellent Python web framework
- pgvector for PostgreSQL vector search capabilities

# Microservices Architecture - Build Time Optimization

## Overview

The application has been split into **two separate services** to dramatically reduce build times:

### 1. **Main Backend** (Port 8001)
- FastAPI application
- Database operations
- Business logic
- **NO ML dependencies** ✅
- **Build time: ~30-60 seconds**

### 2. **Embedding Service** (Port 8002)
- Standalone FastAPI service
- Handles all ML operations (embeddings, reranking)
- Contains PyTorch, sentence-transformers
- **Build time: 3-5 minutes (but only build ONCE)**

## Benefits

✅ **Fast Development**: Main backend builds in seconds
✅ **Separation of Concerns**: ML logic isolated
✅ **Independent Scaling**: Scale embedding service separately
✅ **Easy Testing**: Test backend without ML dependencies

## Build & Run

### First Time Setup (Builds both services)
```bash
# Clean everything
docker-compose down -v
docker system prune -f

# Build all services
DOCKER_BUILDKIT=1 docker-compose up --build
```

**Expected times:**
- Embedding service: 3-5 minutes (FIRST time only)
- Main backend: 30-60 seconds
- Total: ~5 minutes first time

### Subsequent Development (Fast!)
```bash
# Only rebuild backend (if you changed backend code)
docker-compose up --build backend

# This takes only 30-60 seconds! 🚀
```

The embedding service container is **cached** and won't rebuild unless you:
- Change `backend/embedding_service/` files
- Delete the embedding service image

## Service Communication

The backend communicates with the embedding service via HTTP:

```python
# Automatically configured via environment variable
EMBEDDING_SERVICE_URL=http://embedding-service:8002
```

### API Endpoints (Embedding Service)

**Health Check:**
```bash
curl http://localhost:8002/health
```

**Generate Embeddings:**
```bash
curl -X POST http://localhost:8002/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello world", "test document"]}'
```

**Rerank Documents:**
```bash
curl -X POST http://localhost:8002/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "documents": ["AI is cool", "Python programming", "Deep learning"],
    "top_k": 2
  }'
```

## Architecture Diagram

```
┌─────────────────────┐
│   Main Backend      │
│   (Port 8001)       │
│                     │
│   - FastAPI         │
│   - Business Logic  │
│   - NO ML deps      │
│   BUILD: 30s        │
└──────────┬──────────┘
           │ HTTP
           │
┌──────────▼──────────┐
│  Embedding Service  │
│   (Port 8002)       │
│                     │
│   - PyTorch         │
│   - Transformers    │
│   - ML Models       │
│   BUILD: 5min       │
└─────────────────────┘
```

## Files Changed

### New Files
- `backend/embedding_service/main.py` - Embedding microservice
- `backend/embedding_service/requirements.txt` - ML dependencies only
- `backend/embedding_service/Dockerfile` - Separate build

### Modified Files
- `backend/app/services/embedding_service.py` - Now HTTP client
- `backend/app/services/rerank_service.py` - Now HTTP client
- `backend/requirements-app.txt` - Removed ML packages
- `backend/Dockerfile` - Simplified, faster build
- `docker-compose.yml` - Added embedding service

## Troubleshooting

### Embedding service won't start
```bash
# Check logs
docker-compose logs embedding-service

# Rebuild just the embedding service
docker-compose build --no-cache embedding-service
```

### Backend can't connect to embedding service
```bash
# Verify embedding service is healthy
docker-compose ps
curl http://localhost:8002/health

# Check network connectivity
docker-compose exec backend ping embedding-service
```

### Want to rebuild everything from scratch
```bash
docker-compose down -v
docker system prune -af
docker volume prune -f
DOCKER_BUILDKIT=1 docker-compose up --build
```

## Development Workflow

1. **Working on backend code?**
   - Just rebuild backend: `docker-compose up --build backend`
   - Takes 30-60 seconds ✅

2. **Working on ML models?**
   - Rebuild embedding service: `docker-compose up --build embedding-service`
   - Takes 3-5 minutes (rare)

3. **Adding new backend dependencies?**
   - Add to `requirements-app.txt`
   - Rebuild backend (30-60s)

4. **Adding new ML dependencies?**
   - Add to `embedding_service/requirements.txt`
   - Rebuild embedding service (3-5 min)

## Production Considerations

- **Scaling**: You can run multiple embedding service replicas
- **Caching**: Consider adding Redis cache for embeddings
- **Monitoring**: Both services expose `/health` endpoints
- **Load Balancing**: Add nginx for embedding service if needed

---

**Build times before:** 30+ minutes ❌
**Build times now:** 30-60 seconds (for backend) ✅

Enjoy fast builds! 🚀

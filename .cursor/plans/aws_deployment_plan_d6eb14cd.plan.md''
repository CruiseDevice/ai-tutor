---
name: AWS Deployment Plan
overview: Create a cost-effective deployment plan for StudyFetch AI Tutor on AWS using managed services, optimized for learning purposes with free tier and low-cost options.
todos: []
---

# AWS Deployment Plan for StudyFetch AI Tutor

## Overview

This plan deploys the StudyFetch AI Tutor application to AWS using managed services, optimized for cost-effectiveness and learning. The deployment uses AWS free tier where possible and low-cost options for production workloads.

## Scalable Architecture

```mermaid
graph TB
    subgraph "AWS Cloud - Multi-AZ Deployment"
        subgraph "Frontend Layer"
            CloudFront[CloudFront CDN<br/>Global Distribution]
            WAF[WAF Rules]
            S3Frontend[S3 Static Hosting]
        end

        subgraph "Application Layer - Auto-Scaling"
            ALB[Application Load Balancer<br/>Multi-AZ]
            subgraph "Backend API - Auto-Scaling"
                ECS1[ECS Fargate<br/>Backend Task 1]
                ECS2[ECS Fargate<br/>Backend Task 2]
                ECSN[ECS Fargate<br/>Backend Task N]
            end
            subgraph "Embedding Service - Auto-Scaling"
                EMB1[ECS Fargate<br/>Embedding Task 1]
                EMB2[ECS Fargate<br/>Embedding Task 2]
            end
            subgraph "Worker Pool - Auto-Scaling"
                WORK1[ECS Fargate<br/>Worker Task 1]
                WORK2[ECS Fargate<br/>Worker Task 2]
                WORKN[ECS Fargate<br/>Worker Task N]
            end
        end

        subgraph "Data Layer - High Availability"
            subgraph "PostgreSQL Cluster"
                RDSPrimary[RDS Primary<br/>Multi-AZ]
                RDSReplica1[Read Replica 1]
                RDSReplica2[Read Replica 2]
                PgBouncer[PgBouncer<br/>Connection Pooler]
            end
            subgraph "Redis Cluster"
                RedisPrimary[ElastiCache Primary<br/>Cluster Mode]
                RedisReplica1[Replica Node 1]
                RedisReplica2[Replica Node 2]
            end
            S3PDFs[S3 PDF Storage<br/>Lifecycle Policies]
        end

        subgraph "Queue & Messaging"
            SQS[SQS Queue<br/>Document Processing]
            EventBridge[EventBridge<br/>Scheduled Tasks]
        end

        subgraph "Monitoring & Scaling"
            CloudWatch[CloudWatch<br/>Metrics & Logs]
            AutoScaling[Auto Scaling<br/>Policies]
            ASG[Application Auto Scaling]
        end
    end

    User[Users] --> CloudFront
    CloudFront --> WAF
    WAF --> S3Frontend
    S3Frontend --> ALB
    ALB --> ECS1
    ALB --> ECS2
    ALB --> ECSN
    ECS1 --> PgBouncer
    ECS2 --> PgBouncer
    ECSN --> PgBouncer
    PgBouncer --> RDSPrimary
    PgBouncer --> RDSReplica1
    PgBouncer --> RDSReplica2
    ECS1 --> RedisPrimary
    ECS2 --> RedisPrimary
    ECSN --> RedisPrimary
    ECS1 --> EMB1
    ECS2 --> EMB2
    ECSN --> EMB1
    ECS1 --> S3PDFs
    ECS2 --> S3PDFs
    ECSN --> S3PDFs
    ECS1 --> SQS
    WORK1 --> SQS
    WORK2 --> SQS
    WORKN --> SQS
    WORK1 --> RDSPrimary
    WORK2 --> RDSPrimary
    WORKN --> RDSPrimary
    WORK1 --> RedisPrimary
    WORK2 --> RedisPrimary
    WORKN --> RedisPrimary
    WORK1 --> EMB1
    WORK2 --> EMB2
    WORKN --> EMB1
    WORK1 --> S3PDFs
    WORK2 --> S3PDFs
    WORKN --> S3PDFs
    CloudWatch --> AutoScaling
    AutoScaling --> ASG
    ASG --> ECS1
    ASG --> ECS2
    ASG --> ECSN
    ASG --> EMB1
    ASG --> EMB2
    ASG --> WORK1
    ASG --> WORK2
    ASG --> WORKN
```



## Cost Breakdown (Monthly Estimates)

### Free Tier (First 12 Months)

- **RDS PostgreSQL**: Free tier (db.t3.micro, 20GB storage) - $0
- **ElastiCache Redis**: Free tier (cache.t3.micro) - $0
- **S3**: 5GB storage + requests - ~$0-1
- **CloudFront**: 50GB transfer - ~$0-1
- **ECS Fargate**: 0.25 vCPU, 0.5GB RAM per service - ~$5-10
- **ALB**: ~$16/month (no free tier)
- **Data Transfer**: ~$1-5
- **Total**: ~$25-35/month

### Post Free Tier (Minimal Usage)

- **RDS**: db.t3.small - ~$15-20/month
- **ElastiCache**: cache.t3.small - ~$12-15/month
- **ECS Fargate**: ~$15-25/month
- **ALB**: ~$16/month
- **S3 + CloudFront**: ~$2-5/month
- **Total**: ~$60-80/month

## Deployment Steps

### Phase 1: Infrastructure Setup

#### 1.1 AWS Account & Prerequisites

- Create AWS account (if not exists)
- Install AWS CLI: `brew install awscli` (macOS)
- Configure AWS credentials: `aws configure`
- Install Docker Desktop (for building images)
- Install Terraform (optional, for IaC) or use AWS Console

#### 1.2 Create S3 Buckets

- **PDF Storage Bucket**: `studyfetch-pdfs-{your-name}` (private)
- **Frontend Build Bucket**: `studyfetch-frontend-{your-name}` (public)
- Enable versioning and lifecycle policies for cost optimization

#### 1.3 Set Up VPC and Networking

- Create VPC with public/private subnets (use default VPC for simplicity initially)
- Create security groups:
- `backend-sg`: Allow 8001 from ALB
- `embedding-sg`: Allow 8002 from backend-sg
- `db-sg`: Allow 5432 from backend-sg and worker-sg
- `redis-sg`: Allow 6379 from backend-sg and worker-sg
- `alb-sg`: Allow 80/443 from internet

### Phase 2: Scalable Database Setup

#### 2.1 RDS PostgreSQL Primary (Multi-AZ)

- Create RDS PostgreSQL instance:
- Engine: PostgreSQL 16
- Instance: db.t3.medium (baseline) or db.r6g.large (high traffic)
- **Multi-AZ deployment**: Enabled for high availability
- Storage: 100GB GP3 (auto-scaling enabled, max 1TB)
- Storage type: General Purpose SSD (gp3)
- Backup retention: 7 days (30 days for production)
- Enable automated backups
- Enable Performance Insights
- Enable Enhanced Monitoring (60-second intervals)
- **Subnet group**: Create DB subnet group across 2+ AZs
- **Security group**: Only allow from backend/worker security groups
- **Parameter group**: Create custom parameter group:
    - `max_connections`: 200 (adjust based on instance size)
    - `shared_buffers`: 25% of RAM
    - `effective_cache_size`: 75% of RAM
    - `maintenance_work_mem`: 1GB
    - `checkpoint_completion_target`: 0.9
- Install pgvector extension:
  ```sql
    CREATE EXTENSION IF NOT EXISTS vector;
  ```




- Create connection pooler user (for PgBouncer):
  ```sql
    CREATE USER pooler WITH PASSWORD 'secure-password';
    GRANT CONNECT ON DATABASE studyfetch TO pooler;
  ```




#### 2.2 RDS Read Replicas (Horizontal Scaling)

- Create 2 read replicas:
- Same instance type as primary (or one size smaller)
- **Multi-AZ**: Enabled for each replica
- **Read replica lag monitoring**: Enable CloudWatch alarms
- Use for read-heavy operations (document retrieval, search)
- Update application to use read replicas for:
- Document listing queries
- Conversation history queries
- Search queries (vector similarity)
- Connection string pattern:
  ```javascript
    Primary: postgresql://user:pass@primary-endpoint:5432/studyfetch
    Replica 1: postgresql://user:pass@replica-1-endpoint:5432/studyfetch
    Replica 2: postgresql://user:pass@replica-2-endpoint:5432/studyfetch
  ```




#### 2.3 PgBouncer Connection Pooler

- Deploy PgBouncer as ECS Fargate service:
- **Purpose**: Reduce database connections, improve connection reuse
- **Configuration**: Transaction pooling mode
- **Pool size**: 25 connections per backend task
- **Total connections**: 200-500 (based on traffic)
- Create PgBouncer Dockerfile:
  ```dockerfile
    FROM pgbouncer/pgbouncer:latest
    COPY pgbouncer.ini /etc/pgbouncer/
  ```




- PgBouncer config (`pgbouncer.ini`):
  ```ini
    [databases]
    studyfetch = host=primary-endpoint port=5432 dbname=studyfetch

    [pgbouncer]
    pool_mode = transaction
    max_client_conn = 1000
    default_pool_size = 25
    min_pool_size = 5
    reserve_pool_size = 5
    reserve_pool_timeout = 3
  ```




- Backend connects to PgBouncer instead of RDS directly

#### 2.4 ElastiCache Redis Cluster Mode

- Create ElastiCache Redis cluster:
- **Engine**: Redis 7
- **Mode**: Cluster Mode Enabled (for horizontal scaling)
- **Node type**: cache.t3.medium (baseline) or cache.r6g.large (high traffic)
- **Number of shards**: 3 (baseline) or 5 (high traffic)
- **Replicas per shard**: 1-2 (for high availability)
- **Total nodes**: 6-15 nodes
- **Enable encryption**: In-transit and at-rest
- **Enable auto-failover**: Yes
- **Enable automatic backup**: Daily snapshots
- **Backup window**: Off-peak hours
- **Multi-AZ**: Enabled
- **Security group**: Only allow from backend/worker security groups
- Update Redis client configuration for cluster mode:
- Use Redis cluster client library
- Handle cluster topology changes automatically
- Configure connection pooling

### Phase 3: Container Images

#### 3.1 Build and Push Docker Images

- Update Dockerfiles to support both `linux/amd64` (AWS uses x86):
- Modify `backend/Dockerfile` and `backend/embedding_service/Dockerfile`
- Change `--platform=linux/arm64` to `--platform=linux/amd64` or multi-platform
- Create ECR repositories:
  ```bash
    aws ecr create-repository --repository-name studyfetch-backend
    aws ecr create-repository --repository-name studyfetch-embedding
    aws ecr create-repository --repository-name studyfetch-worker
  ```




- Build and push images:
  ```bash
    # Authenticate
    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

    # Build and push backend
    docker buildx build --platform linux/amd64 -t studyfetch-backend ./backend
    docker tag studyfetch-backend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/studyfetch-backend:latest
    docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/studyfetch-backend:latest

    # Repeat for embedding-service and worker
  ```




### Phase 4: ECS Services Setup

#### 4.1 Create ECS Cluster

- Create Fargate cluster: `studyfetch-cluster`
- No EC2 instances needed (serverless)

#### 4.2 Create Task Definitions

- **Backend Task**:
- CPU: 0.25 vCPU (256), Memory: 0.5 GB (512 MB)
- Container: Backend API on port 8001
- Environment variables from AWS Secrets Manager or Parameter Store
- Health check: `/health` endpoint
- **Embedding Service Task**:
- CPU: 0.5 vCPU (512), Memory: 1 GB (ML model needs more)
- Container: Embedding service on port 8002
- Health check: `/health` endpoint
- **Worker Task**:
- CPU: 0.25 vCPU (256), Memory: 0.5 GB (512 MB)
- Container: ARQ worker
- Same environment as backend

#### 4.3 Create ECS Services

- Backend service: 1-2 tasks (for redundancy)
- Embedding service: 1 task (can scale if needed)
- Worker service: 1 task

### Phase 5: Load Balancer & Frontend

#### 5.1 Application Load Balancer

- Create ALB in public subnets
- Target group for backend (port 8001)
- Health check path: `/health`
- Listener on port 80 (add HTTPS later with ACM certificate)

#### 5.2 Frontend Deployment

- Build Next.js app:
  ```bash
    npm run build
    npm run export  # or configure static export
  ```




- Upload to S3 bucket:
  ```bash
    aws s3 sync out/ s3://studyfetch-frontend-{your-name} --delete
  ```




- Configure S3 bucket for static website hosting
- Create CloudFront distribution:
- Origin: S3 bucket
- Default root object: `index.html`
- Enable compression
- Set up custom domain (optional, requires Route53)

### Phase 6: Environment Configuration

#### 6.1 AWS Secrets Manager / Parameter Store

Store sensitive values:

- `JWT_SECRET`
- `ENCRYPTION_KEY`
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (for S3 access)
- `OPENAI_API_KEY` (user-provided, but store admin key)
- Database credentials

#### 6.2 Environment Variables for ECS Tasks

Configure in task definitions:

- `DATABASE_URL`: `postgresql://user:pass@rds-endpoint:5432/studyfetch`
- `REDIS_URL`: `redis://elasticache-endpoint:6379/0`
- `EMBEDDING_SERVICE_URL`: `http://embedding-service:8002` (internal)
- `S3_PDFBUCKET_NAME`: Your S3 bucket name
- `AWS_REGION`: `us-east-1`
- `NODE_ENV`: `production`
- `CORS_ORIGINS`: Your CloudFront/Frontend URL

### Phase 7: Monitoring & Logging

#### 7.1 CloudWatch

- Enable CloudWatch Logs for ECS tasks
- Create log groups:
- `/ecs/studyfetch-backend`
- `/ecs/studyfetch-embedding`
- `/ecs/studyfetch-worker`
- Set up basic alarms for:
- High CPU usage
- Memory usage
- Task failures

#### 7.2 Cost Monitoring

- Set up AWS Cost Explorer
- Create budget alerts (e.g., $50/month)
- Enable detailed billing

## Files to Create/Modify

### New Files

1. `deployment/aws/ecs-task-definitions/backend.json` - Backend task definition
2. `deployment/aws/ecs-task-definitions/embedding.json` - Embedding service task definition
3. `deployment/aws/ecs-task-definitions/worker.json` - Worker task definition
4. `deployment/aws/scripts/build-and-push.sh` - Build and push script
5. `deployment/aws/scripts/deploy-frontend.sh` - Frontend deployment script
6. `deployment/aws/terraform/` (optional) - Infrastructure as Code
7. `deployment/aws/README.md` - Deployment guide

### Files to Modify

1. `backend/Dockerfile` - Change platform to `linux/amd64` or multi-platform
2. `backend/embedding_service/Dockerfile` - Change platform to `linux/amd64`
3. `docker-compose.yml` - Add production override or create `docker-compose.prod.yml`
4. `.env.example` - Add AWS-specific environment variables
5. `next.config.ts` - Configure for static export or SSR deployment

## Learning Resources

1. **AWS Documentation**:

- ECS Fargate: https://docs.aws.amazon.com/ecs/
- RDS PostgreSQL: https://docs.aws.amazon.com/rds/
- ElastiCache: https://docs.aws.amazon.com/elasticache/

2. **Cost Optimization Tips**:

- Use Reserved Instances after free tier (if committing)
- Enable S3 lifecycle policies for old PDFs
- Use CloudFront caching aggressively
- Scale down services during low usage periods
- Monitor and optimize ECS task sizes

3. **Security Best Practices**:

- Move to private subnets after initial setup
- Use IAM roles instead of access keys
- Enable VPC endpoints for S3
- Set up WAF on ALB
- Enable SSL/TLS (ACM certificate)

## Alternative: Simplified Single-Server Deployment

For even lower cost (~$5-10/month), consider:

- Single EC2 t3.micro instance (free tier)
- Run all services via Docker Compose
- Use managed RDS and ElastiCache (still managed)
- Use CloudFlare for CDN (free tier)

## Next Steps After Deployment

1. Set up CI/CD pipeline (GitHub Actions → ECR → ECS)
2. Add HTTPS with ACM certificate
3. Set up custom domain with Route53
4. Implement auto-scaling policies
5. Add database backups and disaster recovery
6. Set up monitoring dashboards

## Estimated Timeline

- **Setup & Learning**: 4-8 hours
- **Initial Deployment**: 2-4 hours
- **Testing & Debugging**: 2-4 hours
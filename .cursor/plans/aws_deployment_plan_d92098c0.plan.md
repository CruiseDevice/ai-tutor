---
name: AWS Deployment Plan
overview: Deploy StudyFetch AI Tutor to AWS with a cost-optimized, scalable architecture suitable for learning and interview discussions. The plan includes containerized services on ECS Fargate, managed databases, CDN, and CI/CD pipeline.
todos:
  - id: infra-vpc
    content: Create VPC with public/private subnets, NAT Gateway, Internet Gateway, and route tables
    status: pending
  - id: infra-security
    content: Create security groups for ALB, ECS services, RDS, and ElastiCache with appropriate rules
    status: pending
    dependencies:
      - infra-vpc
  - id: data-rds
    content: Set up RDS PostgreSQL instance with pgvector extension in private subnet
    status: pending
    dependencies:
      - infra-vpc
      - infra-security
  - id: data-redis
    content: Set up ElastiCache Redis cluster in private subnet
    status: pending
    dependencies:
      - infra-vpc
      - infra-security
  - id: data-s3
    content: Create S3 buckets for PDF storage and frontend static files with appropriate policies
    status: pending
  - id: container-ecr
    content: Create ECR repositories for backend-api, embedding-service, and worker
    status: pending
  - id: container-dockerfiles
    content: Update Dockerfiles for production (remove --reload, change platform to linux/amd64, optimize)
    status: pending
  - id: container-task-defs
    content: Create ECS task definitions for all three services with proper resource allocation and secrets
    status: pending
    dependencies:
      - container-ecr
      - data-rds
      - data-redis
      - data-s3
  - id: networking-alb
    content: Create Application Load Balancer with target groups and listener rules
    status: pending
    dependencies:
      - infra-vpc
      - infra-security
  - id: ecs-cluster
    content: Create ECS Fargate cluster and deploy all three services
    status: pending
    dependencies:
      - networking-alb
      - container-task-defs
  - id: frontend-build
    content: Configure Next.js for static export and update next.config.ts
    status: pending
  - id: frontend-deploy
    content: Set up S3 bucket for frontend, CloudFront distribution, and Lambda@Edge for routing
    status: pending
    dependencies:
      - frontend-build
      - data-s3
  - id: secrets-setup
    content: Store all sensitive configuration in AWS Secrets Manager
    status: pending
  - id: monitoring
    content: Set up CloudWatch log groups, metrics, alarms, and dashboards
    status: pending
    dependencies:
      - ecs-cluster
  - id: cicd-pipeline
    content: Create GitHub Actions workflow for automated build, push to ECR, and ECS deployment
    status: pending
    dependencies:
      - container-ecr
      - ecs-cluster
  - id: domain-ssl
    content: Configure Route 53 hosted zone, ACM certificates, and domain routing
    status: pending
    dependencies:
      - frontend-deploy
      - networking-alb
---

# AWS Deployment Plan for StudyFetch AI Tutor

## Architecture Overview

This plan deploys a scalable, cost-optimized AWS architecture suitable for learning and interview discussions. The architecture separates concerns and uses managed AWS services for reliability and scalability.

### High-Level Architecture

```mermaid
graph TB
    subgraph Internet["Internet"]
        User[Users]
    end

    subgraph CDN["CloudFront CDN"]
        CF[CloudFront Distribution]
    end

    subgraph Frontend["Frontend Layer"]
        S3Frontend[S3 Bucket<br/>Static Next.js Build]
        LambdaEdge[Lambda@Edge<br/>Routing/Headers]
    end

    subgraph ALB["Application Load Balancer"]
        ALBInstance[ALB<br/>HTTPS/TLS]
    end

    subgraph ECS["ECS Fargate Cluster"]
        BackendAPI[Backend API<br/>FastAPI :8001]
        EmbeddingService[Embedding Service<br/>FastAPI :8002]
        Worker[ARQ Worker<br/>Background Jobs]
    end

    subgraph Data["Data Layer"]
        RDS[(RDS PostgreSQL<br/>with pgvector)]
        ElastiCache[(ElastiCache Redis<br/>Cache & Queue)]
        S3Storage[S3 Bucket<br/>PDF Storage]
    end

    subgraph AWS["AWS Services"]
        SecretsManager[Secrets Manager<br/>Environment Variables]
        CloudWatch[CloudWatch<br/>Logs & Metrics]
        ECR[ECR<br/>Container Registry]
    end

    User --> CF
    CF --> LambdaEdge
    LambdaEdge --> S3Frontend
    S3Frontend --> ALBInstance
    ALBInstance --> BackendAPI
    ALBInstance --> EmbeddingService
    BackendAPI --> RDS
    BackendAPI --> ElastiCache
    BackendAPI --> S3Storage
    Worker --> RDS
    Worker --> ElastiCache
    Worker --> S3Storage
    EmbeddingService --> CloudWatch
    BackendAPI --> SecretsManager
    BackendAPI --> CloudWatch
```



## Components Breakdown

### 1. Frontend Deployment

- **Next.js Static Export** → S3 + CloudFront
- **Alternative**: Vercel (free tier, easier setup)
- **Domain**: Route 53 + ACM certificate for HTTPS

### 2. Backend Services (ECS Fargate)

- **Backend API**: FastAPI service (port 8001)
- **Embedding Service**: ML microservice (port 8002)
- **Worker**: ARQ background worker
- All services run as separate ECS tasks in Fargate

### 3. Data Layer

- **RDS PostgreSQL**: Managed PostgreSQL with pgvector extension
- **ElastiCache Redis**: Managed Redis for caching and job queue
- **S3**: PDF document storage

### 4. Networking

- **VPC**: Custom VPC with public/private subnets
- **ALB**: Application Load Balancer for backend services
- **Security Groups**: Restrictive firewall rules

### 5. CI/CD Pipeline

- **GitHub Actions**: Automated build and deployment
- **ECR**: Container registry for Docker images
- **Automated deployments** on push to main branch

## Implementation Steps

### Phase 1: Infrastructure Setup

#### 1.1 VPC and Networking

- Create VPC with CIDR `10.0.0.0/16`
- Create 2 public subnets (multi-AZ) for ALB
- Create 2 private subnets (multi-AZ) for ECS tasks
- Create NAT Gateway (cost-optimized: single NAT in one AZ)
- Create Internet Gateway for public subnets
- Configure route tables

#### 1.2 Security Groups

- **ALB Security Group**: Allow HTTPS (443) from CloudFront/Internet
- **ECS Backend SG**: Allow traffic from ALB on port 8001
- **ECS Embedding SG**: Allow traffic from ALB on port 8002
- **ECS Worker SG**: No inbound, outbound to RDS/Redis/S3
- **RDS Security Group**: Allow PostgreSQL (5432) from ECS SGs
- **Redis Security Group**: Allow Redis (6379) from ECS SGs

### Phase 2: Data Layer Setup

#### 2.1 RDS PostgreSQL

- **Instance**: `db.t3.micro` or `db.t3.small` (cost-optimized)
- **Engine**: PostgreSQL 16
- **Storage**: 20GB GP3 (auto-scaling enabled)
- **Multi-AZ**: Disabled initially (enable for production)
- **Backups**: 7-day retention
- **Extensions**: Enable `pgvector` extension
- **Parameter Group**: Custom group for pgvector settings

#### 2.2 ElastiCache Redis

- **Node Type**: `cache.t3.micro` or `cache.t3.small`
- **Engine**: Redis 7.x
- **Multi-AZ**: Disabled initially
- **Backup**: Daily snapshots
- **Subnet Group**: Private subnets

#### 2.3 S3 Buckets

- **PDF Storage Bucket**: Private bucket for document storage
- **Frontend Bucket**: Public bucket for static Next.js build
- **Enable versioning** and **lifecycle policies** for cost optimization

### Phase 3: Container Setup

#### 3.1 ECR Repositories

Create three ECR repositories:

- `studyfetch/backend-api`
- `studyfetch/embedding-service`
- `studyfetch/worker`

#### 3.2 Docker Image Updates

- Update Dockerfiles to use `linux/amd64` platform (AWS uses x86_64)
- Remove `--reload` flag from production CMD
- Add healthcheck endpoints
- Optimize image sizes (multi-stage builds)

#### 3.3 ECS Task Definitions

Create three task definitions:

- **Backend API Task**: 1 vCPU, 2GB RAM, port 8001
- **Embedding Service Task**: 2 vCPU, 4GB RAM, port 8002
- **Worker Task**: 1 vCPU, 2GB RAM, no exposed ports

### Phase 4: Application Load Balancer

#### 4.1 ALB Configuration

- **Type**: Application Load Balancer (Internet-facing)
- **Scheme**: Internet-facing
- **Subnets**: Public subnets (multi-AZ)
- **Security Group**: Allow HTTPS from CloudFront/Internet

#### 4.2 Target Groups

- **Backend API TG**: Port 8001, health check `/health`
- **Embedding Service TG**: Port 8002, health check `/health`
- **Health Check**: HTTP, path `/health`, interval 30s

#### 4.3 Listener Rules

- **Default Rule**: Route to Backend API TG
- **Path-based Rule**: `/embedding/*` → Embedding Service TG

### Phase 5: ECS Cluster and Services

#### 5.1 ECS Cluster

- **Launch Type**: Fargate (serverless)
- **Cluster Name**: `studyfetch-cluster`

#### 5.2 ECS Services

- **Backend API Service**:
- Desired count: 2 (for high availability)
- Auto-scaling: Scale based on CPU/memory
- **Embedding Service Service**:
- Desired count: 1 (can scale based on load)
- **Worker Service**:
- Desired count: 1 (can scale based on queue depth)

### Phase 6: Frontend Deployment

#### 6.1 Next.js Build Configuration

- Update `next.config.ts` for static export
- Set `NEXT_PUBLIC_BACKEND_URL` to ALB URL
- Build static export: `npm run build && npm run export`

#### 6.2 S3 + CloudFront Setup

- Upload static build to S3 bucket
- Create CloudFront distribution
- Configure origin: S3 bucket
- Add Lambda@Edge for routing (SPA support)
- Configure custom domain with ACM certificate

### Phase 7: Secrets and Configuration

#### 7.1 AWS Secrets Manager

Store sensitive configuration:

- Database credentials
- JWT secret
- Encryption keys
- AWS credentials (for S3 access)
- OpenAI API keys (user-provided, but store encryption key)

#### 7.2 Environment Variables

ECS tasks will pull from:

- Secrets Manager (sensitive values)
- ECS Task Definition (non-sensitive config)
- Parameter Store (optional, for non-sensitive config)

### Phase 8: Monitoring and Logging

#### 8.1 CloudWatch

- **Log Groups**: One per service (backend-api, embedding-service, worker)
- **Metrics**: CPU, memory, request count, error rate
- **Alarms**: High CPU, memory, error rate thresholds
- **Dashboards**: Custom dashboard for service health

#### 8.2 Application Logs

- Configure ECS tasks to send logs to CloudWatch
- Use structured logging (JSON format)
- Set log retention: 7 days (cost optimization)

### Phase 9: CI/CD Pipeline

#### 9.1 GitHub Actions Workflow

Create `.github/workflows/deploy.yml`:

1. **Build**: Build Docker images for all services
2. **Test**: Run tests (optional)
3. **Push**: Push images to ECR
4. **Deploy**: Update ECS services with new images
5. **Frontend**: Build and deploy Next.js to S3

#### 9.2 Deployment Strategy

- **Blue/Green**: Use ECS blue/green deployments
- **Rollback**: Keep previous task definition for quick rollback
- **Health Checks**: Wait for health checks before routing traffic

### Phase 10: Domain and SSL

#### 10.1 Route 53

- Create hosted zone for your domain
- Create A record (alias) pointing to CloudFront distribution
- Create CNAME for www subdomain (optional)

#### 10.2 ACM Certificates

- Request certificate for your domain (us-east-1 for CloudFront)
- Request certificate for ALB (same region as ALB)
- Validate certificates via DNS

## Cost Optimization Strategies

1. **RDS**: Use `db.t3.micro`, disable Multi-AZ initially
2. **ElastiCache**: Use `cache.t3.micro`
3. **ECS Fargate**: Right-size tasks, use spot capacity for workers (optional)
4. **NAT Gateway**: Single NAT in one AZ (not multi-AZ)
5. **S3**: Use Intelligent-Tiering storage class
6. **CloudWatch**: 7-day log retention, delete old logs
7. **ALB**: Use single ALB for all backend services
8. **Reserved Capacity**: Consider Reserved Instances for RDS if running 24/7

## Estimated Monthly Costs (Learning/Development)

- **RDS PostgreSQL** (db.t3.micro): ~$15-20/month
- **ElastiCache Redis** (cache.t3.micro): ~$12-15/month
- **ECS Fargate** (2 backend + 1 embedding + 1 worker): ~$30-50/month
- **ALB**: ~$16/month
- **NAT Gateway**: ~$32/month (largest cost)
- **S3**: ~$5-10/month (depends on storage)
- **CloudFront**: ~$1-5/month (depends on traffic)
- **Route 53**: ~$0.50/month
- **Secrets Manager**: ~$0.40/month
- **CloudWatch**: ~$5-10/month
- **Data Transfer**: ~$5-15/month

**Total**: ~$120-180/month (can be reduced with spot instances, single-AZ, etc.)

## Files to Create/Modify

### New Files

- `infrastructure/terraform/main.tf` - Terraform configuration (optional)
- `infrastructure/cloudformation/template.yaml` - CloudFormation template (optional)
- `.github/workflows/deploy.yml` - CI/CD pipeline
- `infrastructure/ecs/backend-task-definition.json` - ECS task definition
- `infrastructure/ecs/embedding-task-definition.json` - ECS task definition
- `infrastructure/ecs/worker-task-definition.json` - ECS task definition
- `infrastructure/scripts/deploy.sh` - Deployment script
- `infrastructure/scripts/setup-secrets.sh` - Secrets setup script
- `docker-compose.prod.yml` - Production Docker Compose (for reference)

### Modified Files

- `backend/Dockerfile` - Update for production (remove --reload, change platform)
- `backend/embedding_service/Dockerfile` - Update for production
- `next.config.ts` - Add static export configuration
- `src/lib/api-client.ts` - Already uses env var (no changes needed)
- `.env.example` - Add AWS-specific environment variables

## Interview Talking Points

1. **Microservices Architecture**: Separated backend API, embedding service, and workers for independent scaling
2. **Container Orchestration**: ECS Fargate for serverless container management
3. **High Availability**: Multi-AZ deployment, auto-scaling, health checks
4. **Security**: VPC isolation, security groups, Secrets Manager, HTTPS everywhere
5. **Scalability**: Auto-scaling based on CPU/memory, ALB for load distribution
6. **Cost Optimization**: Right-sized instances, single-AZ for non-critical services, intelligent S3 tiering
7. **CI/CD**: Automated deployments via GitHub Actions
8. **Monitoring**: CloudWatch for logs, metrics, and alarms
9. **Disaster Recovery**: Automated backups, multi-AZ capability (can enable)
10. **Performance**: CloudFront CDN, Redis caching, connection pooling

## Next Steps After Deployment

1. Set up monitoring dashboards
2. Configure auto-scaling policies
3. Set up alerting for critical metrics
4. Document runbooks for common operations
5. Set up backup and restore procedures
6. Configure cost alerts
7. Enable AWS Cost Explorer for tracking
8. Consider AWS WAF for additional security
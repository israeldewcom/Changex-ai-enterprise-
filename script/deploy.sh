#!/bin/bash

set -e

echo "🚀 Starting Enhanced ChangeX Enterprise AI Platform Deployment..."

# Environment variables
ENVIRONMENT=${1:-production}
APP_NAME="changex-enterprise-ai"
VERSION="7.0.0"
DOCKER_REGISTRY="registry.changex-ai.com"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validation functions
validate_environment() {
    log_info "Validating deployment environment..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check environment file
    if [ ! -f ".env.${ENVIRONMENT}" ]; then
        log_error "Environment file .env.${ENVIRONMENT} not found"
        exit 1
    fi
    
    log_info "Environment validation passed"
}

build_images() {
    log_info "Building Docker images..."
    
    # Build application image
    docker build -t ${DOCKER_REGISTRY}/${APP_NAME}:${VERSION} .
    docker build -t ${DOCKER_REGISTRY}/${APP_NAME}:latest .
    
    # Build additional service images if needed
    log_info "Docker images built successfully"
}

push_images() {
    log_info "Pushing images to registry..."
    
    # Push to container registry
    docker push ${DOCKER_REGISTRY}/${APP_NAME}:${VERSION}
    docker push ${DOCKER_REGISTRY}/${APP_NAME}:latest
    
    log_info "Images pushed to registry successfully"
}

deploy_infrastructure() {
    log_info "Deploying infrastructure..."
    
    # Copy environment file
    cp .env.${ENVIRONMENT} .env
    
    # Deploy database and core services
    docker-compose up -d postgres redis
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    sleep 30
    
    # Run database migrations
    log_info "Running database migrations..."
    docker-compose run --rm changex-ai python -m alembic upgrade head
    
    # Deploy application
    docker-compose up -d changex-ai
    
    # Deploy monitoring
    docker-compose up -d nginx jaeger prometheus grafana
    
    log_info "Infrastructure deployment completed"
}

run_health_checks() {
    log_info "Running health checks..."
    
    # Wait for application to start
    sleep 30
    
    # Check application health
    if curl -f http://localhost:8000/health; then
        log_info "Application health check passed"
    else
        log_error "Application health check failed"
        exit 1
    fi
    
    # Check database connectivity
    if docker-compose exec postgres pg_isready -U changex; then
        log_info "Database health check passed"
    else
        log_error "Database health check failed"
        exit 1
    fi
    
    log_info "All health checks passed"
}

run_security_scan() {
    log_info "Running security scan..."
    
    # Run vulnerability scanning
    if command -v trivy &> /dev/null; then
        trivy image ${DOCKER_REGISTRY}/${APP_NAME}:${VERSION}
    else
        log_warn "Trivy not installed, skipping vulnerability scan"
    fi
    
    log_info "Security scan completed"
}

# Main deployment process
main() {
    log_info "Starting deployment for ${ENVIRONMENT} environment"
    
    validate_environment
    build_images
    push_images
    deploy_infrastructure
    run_health_checks
    run_security_scan
    
    log_info "🎉 Enhanced ChangeX Enterprise AI Platform deployed successfully!"
    log_info "📊 Application URL: http://localhost:8000"
    log_info "📈 Monitoring URL: http://localhost:3000"
    log_info "🔍 Tracing URL: http://localhost:16686"
}

# Run main function
main "$@"

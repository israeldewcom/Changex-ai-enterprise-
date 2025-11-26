#!/bin/bash

echo "🚀 Setting up FREE GitHub Deployment..."

# Create necessary directories
mkdir -p .github/workflows

# Copy configuration files (you'll create these)
cp deploy-configs/deploy.yml .github/workflows/
cp deploy-configs/docker-compose.prod.yml .
cp deploy-configs/render.yaml .
cp deploy-configs/railway.json .
cp deploy-configs/fly.toml .

# Make scripts executable
chmod +x scripts/*.sh

echo "✅ Deployment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Create accounts on: Render.com, Railway.app, Fly.io"
echo "2. Set up GitHub Secrets with your API keys"
echo "3. Push to main branch to trigger deployment"
echo "4. Your app will deploy to 3+ platforms automatically!"

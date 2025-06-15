#!/bin/bash

echo "Starting MONET Dermatology Analyzer with Docker..."

# Build and start all services
docker-compose up -d

echo ""
echo "Services starting..."
echo "Backend API will be available at: http://localhost:8000"
echo "Frontend will be available at: http://localhost:3000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f backend"
echo "  docker-compose logs -f frontend"
echo ""
echo "To stop all services:"
echo "  docker-compose down" 
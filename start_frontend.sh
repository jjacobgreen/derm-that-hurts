#!/bin/bash

echo "Starting MONET Dermatology Analyzer Frontend..."

cd frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start React development server
echo "Starting React development server on http://localhost:3000"
npm run dev 
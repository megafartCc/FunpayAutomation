#!/bin/bash
# Build script for Railway deployment
# Builds the frontend and ensures static files are ready

echo "Building frontend..."
cd frontend
npm install
npm run build
cd ..

echo "Frontend build complete. Static files are in ./static/"
echo "Starting backend server..."

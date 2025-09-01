#!/bin/bash

echo "🚀 Starting React Frontend..."

cd frontend

# Check if node_modules exists, if not install dependencies
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

# Start the React development server
echo "🌐 Starting React app on http://localhost:3000"
npm start

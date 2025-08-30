#!/bin/bash

echo "🎭 Starting ShakespeareGPT..."

# Check if model files exist
if [ ! -f "checkpoint.pt" ]; then
    echo "❌ Error: checkpoint.pt not found!"
    echo "Please copy it from your ShakespeareLLM folder:"
    echo "cp ../ShakespeareLLM/checkpoint.pt ./"
    exit 1
fi

if [ ! -f "train.txt" ]; then
    echo "❌ Error: train.txt not found!"
    echo "Please copy it from your ShakespeareLLM folder:"
    echo "cp ../ShakespeareLLM/train.txt ./"
    exit 1
fi

echo "✅ Model files found"

# Start backend
echo "🚀 Starting backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://user:password@localhost/shakespearegpt"
export CHECKPOINT_PATH="../checkpoint.pt"
export TRAIN_TEXT_PATH="../train.txt"

# Start backend in background
echo "🌐 Backend starting on http://localhost:8000"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "🎨 Starting frontend..."
cd frontend

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📥 Installing Node.js dependencies..."
    npm install
fi

# Start frontend
echo "🌐 Frontend starting on http://localhost:3000"
npm run dev &
FRONTEND_PID=$!

cd ..

echo ""
echo "🎉 ShakespeareGPT is starting up!"
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend:  http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for user to stop
trap "echo ''; echo '🛑 Stopping ShakespeareGPT...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT

# Keep script running
wait

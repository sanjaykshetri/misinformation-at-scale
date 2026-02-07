#!/bin/bash
# Quick start script for Streamlit dashboard

echo "🚀 Starting Streamlit Misinformation Dashboard..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9+"
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip not found"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "streamlit_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv streamlit_env
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source streamlit_env/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r dashboard_requirements.txt

# Run Streamlit app
echo "✅ Starting dashboard..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Streamlit Dashboard is running!"
echo "📊 Access it at: http://localhost:8501"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

streamlit run app.py

# Cleanup on exit
deactivate

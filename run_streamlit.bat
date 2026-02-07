@echo off
REM Quick start script for Streamlit dashboard on Windows

echo.
echo 🚀 Starting Streamlit Misinformation Dashboard...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "streamlit_env" (
    echo 📦 Creating virtual environment...
    python -m venv streamlit_env
)

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call streamlit_env\Scripts\activate.bat

REM Install/upgrade dependencies
echo 📥 Installing dependencies...
python -m pip install -q --upgrade pip
pip install -q -r dashboard_requirements.txt

REM Run Streamlit app
echo ✅ Starting dashboard...
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 🎯 Streamlit Dashboard is running!
echo 📊 Access it at: http://localhost:8501
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

python -m streamlit run app.py

REM Deactivate virtual environment
deactivate

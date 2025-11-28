@echo off
echo ===================================================
echo Starting Multilingual Meeting Summarizer...
echo ===================================================
echo.
echo Step 1: Checking for installed dependencies...
pip install -r requirements.txt
echo.
echo Step 2: Starting the Server...
echo.
echo IMPORTANT: 
echo 1. Wait for the message "Uvicorn running on http://127.0.0.1:8000"
echo 2. Then open your web browser and go to: http://127.0.0.1:8000
echo.
python -m uvicorn app.backend.main:app --reload
pause

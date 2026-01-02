@echo off
echo Starting Tea Traffic Cafe...

echo Starting Backend...
start "Tea Traffic Backend" cmd /k "cd backend && python -m pip install -r requirements.txt && python warmup.py && python -m uvicorn app:app --reload"

echo Starting Frontend...
start "Tea Traffic Frontend" cmd /k "cd frontend && npm install && npm run dev"

echo Done! the App should open in your browser shortly at http://localhost:5173
pause

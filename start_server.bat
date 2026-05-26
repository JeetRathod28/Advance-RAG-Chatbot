@echo off
call conda activate ragadv
cd /d "C:\Users\Hp\Documents\Projects\RAG chatbot"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 > server.log 2>&1

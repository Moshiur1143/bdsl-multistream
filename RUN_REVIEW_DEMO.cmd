@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" app.py --mode fusion_v2 --review-mode
pause

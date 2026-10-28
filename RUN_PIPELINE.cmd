@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -u scripts\run_pipeline.py
pause

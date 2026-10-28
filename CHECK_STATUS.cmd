@echo off
cd /d "%~dp0"
if exist results\v2_status.json (
  type results\v2_status.json
  powershell -NoProfile -Command "Get-Content -LiteralPath 'results\v2_training.log' -Tail 8"
) else (
  if exist results\pipeline_status.json type results\pipeline_status.json
  if exist dataset\download_progress.json type dataset\download_progress.json
  powershell -NoProfile -Command "Get-Content -LiteralPath 'results\pipeline.log' -Tail 8"
)
pause

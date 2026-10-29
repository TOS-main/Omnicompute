@echo off
rem Double-click to install (first time) and start Omnicompute on Windows.
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python is not installed. Install it from https://www.python.org/downloads/ and run this again.
  pause
  exit /b 1
)
py -m pip install --quiet --upgrade .
if errorlevel 1 (
  echo Installing Omnicompute failed, see the messages above.
  pause
  exit /b 1
)
py -m omnicompute %*
pause

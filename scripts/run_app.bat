@echo off
REM Start the CADICA research prototype (Gradio demo).
REM Uses the project 3.14 Python (CUDA torch). NOT a diagnostic device.
cd /d "%~dp0.."
"C:\Users\HP\AppData\Local\Programs\Python\Python314\python.exe" app.py
if errorlevel 1 (
  echo.
  echo [run_app.bat] app.py exited with an error. See the console output above.
  pause
)

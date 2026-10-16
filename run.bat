@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First run: follow the setup commands in README.md to create .venv and install requirements.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py

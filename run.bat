@echo off
echo ======================================
echo Autism Detection Project - Quick Start
echo ======================================
echo.

REM Check if virtual environment exists
if exist venv (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found!
    echo Please run setup.ps1 first
    pause
    exit /b 1
)

echo.
echo Starting Django development server...
echo.
echo The application will be available at: http://127.0.0.1:8000/
echo.
echo Press Ctrl+C to stop the server
echo.

python manage.py runserver

pause

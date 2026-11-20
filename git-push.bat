@echo off
REM Quick Git Push Script for Autism Detection Project
REM Usage: Simply double-click this file or run: git-push.bat

echo ============================================
echo   Autism Detection - Quick Git Push
echo ============================================
echo.

REM Check if git is initialized
if not exist ".git" (
    echo ERROR: Git repository not initialized!
    echo Please run setup first.
    pause
    exit /b 1
)

REM Show current status
echo Current Status:
echo ---------------
git status
echo.

REM Ask for commit message
set /p MESSAGE="Enter commit message (or press Enter for default): "
if "%MESSAGE%"=="" set MESSAGE="chore: update project files"

echo.
echo Adding all changes...
git add .

echo.
echo Committing with message: %MESSAGE%
git commit -m "%MESSAGE%"

echo.
echo Pushing to GitHub...
git push

echo.
echo ============================================
echo   Push Complete!
echo ============================================
echo.
pause

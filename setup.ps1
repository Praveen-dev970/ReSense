# Autism Detection Project - Automated Setup Script for Windows PowerShell

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Autism Detection Project Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists." -ForegroundColor Yellow
    $createVenv = Read-Host "Do you want to recreate it? (y/n)"
    if ($createVenv -eq "y") {
        Remove-Item -Recurse -Force venv
        Write-Host "Creating virtual environment..." -ForegroundColor Green
        python -m venv venv
    }
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Green
    python -m venv venv
}

Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

Write-Host ""
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host ""
Write-Host "Creating necessary directories..." -ForegroundColor Green
New-Item -ItemType Directory -Force -Path "static\uploads" | Out-Null
New-Item -ItemType Directory -Force -Path "output_frames" | Out-Null
New-Item -ItemType Directory -Force -Path "dataset\training_set\Autistic" | Out-Null
New-Item -ItemType Directory -Force -Path "dataset\training_set\Non_Autistic" | Out-Null
New-Item -ItemType Directory -Force -Path "dataset\test_set\Autistic" | Out-Null
New-Item -ItemType Directory -Force -Path "dataset\test_set\Non_Autistic" | Out-Null

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Setup MySQL database (create 'autism' database)" -ForegroundColor White
Write-Host "2. Import database schema from database/autism.sql" -ForegroundColor White
Write-Host "3. Update database credentials in textutils/settings.py if needed" -ForegroundColor White
Write-Host "4. Run: python manage.py migrate" -ForegroundColor White
Write-Host "5. Run: python manage.py runserver" -ForegroundColor White
Write-Host ""
Write-Host "To activate the virtual environment in the future, run:" -ForegroundColor Yellow
Write-Host ".\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""

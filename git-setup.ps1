# Git Setup and Push Script for Autism Detection System (Windows PowerShell)
# This script helps you properly set up and push your project to GitHub

Write-Host "===================================" -ForegroundColor Cyan
Write-Host "Autism Detection System - Git Setup" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
try {
    $gitVersion = git --version
    Write-Host "✓ Git is installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git is not installed. Please install Git first." -ForegroundColor Red
    Write-Host "Download from: https://git-scm.com/download/win"
    exit 1
}

# Check if git is initialized
if (-Not (Test-Path ".git")) {
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✓ Git initialized" -ForegroundColor Green
} else {
    Write-Host "✓ Git repository already initialized" -ForegroundColor Green
}

# Check current remote
try {
    $currentRemote = git remote get-url origin 2>$null
} catch {
    $currentRemote = $null
}

if (-Not $currentRemote) {
    Write-Host ""
    Write-Host "Setting up GitHub remote..." -ForegroundColor Yellow
    Write-Host "Default: https://github.com/Venu22003/Autism-detection-system.git" -ForegroundColor Cyan
    $repoUrl = Read-Host "Press Enter to use default, or enter your repository URL"
    
    if ([string]::IsNullOrWhiteSpace($repoUrl)) {
        $repoUrl = "https://github.com/Venu22003/Autism-detection-system.git"
    }
    
    git remote add origin $repoUrl
    Write-Host "✓ Remote added: $repoUrl" -ForegroundColor Green
} else {
    Write-Host "✓ Remote already configured: $currentRemote" -ForegroundColor Green
}

# Check for uncommitted changes
$status = git status --porcelain

if ($status) {
    Write-Host ""
    Write-Host "Files to be committed:" -ForegroundColor Yellow
    git status --short
    Write-Host ""
    $commitMsg = Read-Host "Enter commit message (or press Enter for default)"
    
    if ([string]::IsNullOrWhiteSpace($commitMsg)) {
        $commitMsg = "Initial commit - Autism Detection System with comprehensive documentation"
    }
    
    Write-Host ""
    Write-Host "Adding files to git..." -ForegroundColor Yellow
    git add .
    
    Write-Host "Committing changes..." -ForegroundColor Yellow
    git commit -m $commitMsg
    Write-Host "✓ Changes committed" -ForegroundColor Green
} else {
    Write-Host "✓ No uncommitted changes" -ForegroundColor Green
}

# Check current branch
$branch = git branch --show-current

# Push to GitHub
Write-Host ""
$pushConfirm = Read-Host "Do you want to push to GitHub now? (y/n)"

if ($pushConfirm -eq "y" -or $pushConfirm -eq "Y") {
    Write-Host "Pushing to GitHub (branch: $branch)..." -ForegroundColor Yellow
    
    try {
        git push -u origin $branch 2>&1
        
        Write-Host ""
        Write-Host "✓✓✓ Success! Your project is now on GitHub! ✓✓✓" -ForegroundColor Green
        Write-Host ""
        Write-Host "View your repository at:" -ForegroundColor Cyan
        
        if ($currentRemote) {
            Write-Host $currentRemote -ForegroundColor Cyan
        }
        
        Write-Host ""
        Write-Host "Next Steps:" -ForegroundColor Yellow
        Write-Host "1. Add a repository description on GitHub"
        Write-Host "2. Add topics/tags for better discoverability"
        Write-Host "3. Enable GitHub Pages if you want a project website"
        Write-Host "4. Set up branch protection rules"
        
    } catch {
        Write-Host ""
        Write-Host "⚠ Push failed. Please check your credentials and try again." -ForegroundColor Red
        Write-Host ""
        Write-Host "Tips:" -ForegroundColor Yellow
        Write-Host "1. Make sure you have access to the repository"
        Write-Host "2. Configure your GitHub credentials: git config --global user.name 'Your Name'"
        Write-Host "3. Configure your email: git config --global user.email 'your@email.com'"
        Write-Host "4. If using HTTPS, you may need a Personal Access Token"
        Write-Host "   Generate one at: https://github.com/settings/tokens"
    }
} else {
    Write-Host "Skipped push. You can push later with: git push -u origin $branch" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Useful Git Commands:" -ForegroundColor Yellow
Write-Host "  git status          - Check status of your files"
Write-Host "  git add .           - Stage all changes"
Write-Host "  git commit -m 'msg' - Commit with message"
Write-Host "  git push            - Push to GitHub"
Write-Host "  git pull            - Pull latest changes"
Write-Host "  git log             - View commit history"
Write-Host ""

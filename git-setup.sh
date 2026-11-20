#!/bin/bash
# Git Setup and Push Script for Autism Detection System
# This script helps you properly set up and push your project to GitHub

echo "==================================="
echo "Autism Detection System - Git Setup"
echo "==================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    echo "✓ Git initialized"
else
    echo "✓ Git repository already initialized"
fi

# Check current remote
CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null)

if [ -z "$CURRENT_REMOTE" ]; then
    echo ""
    echo "Setting up GitHub remote..."
    echo "Default: https://github.com/Venu22003/Autism-detection-system.git"
    read -p "Press Enter to use default, or enter your repository URL: " REPO_URL
    
    if [ -z "$REPO_URL" ]; then
        REPO_URL="https://github.com/Venu22003/Autism-detection-system.git"
    fi
    
    git remote add origin "$REPO_URL"
    echo "✓ Remote added: $REPO_URL"
else
    echo "✓ Remote already configured: $CURRENT_REMOTE"
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo ""
    echo "Files to be committed:"
    git status --short
    echo ""
    read -p "Enter commit message (or press Enter for default): " COMMIT_MSG
    
    if [ -z "$COMMIT_MSG" ]; then
        COMMIT_MSG="Initial commit - Autism Detection System"
    fi
    
    echo ""
    echo "Adding files to git..."
    git add .
    
    echo "Committing changes..."
    git commit -m "$COMMIT_MSG"
    echo "✓ Changes committed"
else
    echo "✓ No uncommitted changes"
fi

# Push to GitHub
echo ""
read -p "Do you want to push to GitHub now? (y/n): " PUSH_CONFIRM

if [ "$PUSH_CONFIRM" = "y" ] || [ "$PUSH_CONFIRM" = "Y" ]; then
    BRANCH=$(git branch --show-current)
    
    echo "Pushing to GitHub (branch: $BRANCH)..."
    git push -u origin "$BRANCH"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓✓✓ Success! Your project is now on GitHub! ✓✓✓"
        echo ""
        echo "View your repository at:"
        echo "$CURRENT_REMOTE"
    else
        echo ""
        echo "⚠ Push failed. Please check your credentials and try again."
        echo ""
        echo "Tips:"
        echo "1. Make sure you have access to the repository"
        echo "2. Configure your GitHub credentials"
        echo "3. If using HTTPS, you may need a Personal Access Token"
    fi
else
    echo "Skipped push. You can push later with: git push -u origin master"
fi

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="

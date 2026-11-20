# 🚀 GitHub Setup Guide - Autism Detection System

Complete guide to properly setting up and pushing your Autism Detection System to GitHub.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Setup (Automated)](#quick-setup-automated)
- [Manual Setup](#manual-setup)
- [First Time Git Configuration](#first-time-git-configuration)
- [Creating GitHub Repository](#creating-github-repository)
- [Pushing to GitHub](#pushing-to-github)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Prerequisites

Before you begin, ensure you have:
- ✅ Git installed on your system
- ✅ GitHub account created
- ✅ Basic understanding of Git commands
- ✅ Terminal/PowerShell access

### Check Git Installation

```bash
# Check if Git is installed
git --version

# If not installed, download from:
# https://git-scm.com/download
```

## Quick Setup (Automated)

### Option 1: Using PowerShell Script (Windows - Recommended)

```powershell
# Navigate to project directory
cd "d:\projects\Capstone_Mine\Python-Autism Detection\Python-Autism Detection"

# Run the automated setup script
.\git-setup.ps1
```

### Option 2: Using Bash Script (Linux/Mac)

```bash
# Navigate to project directory
cd ~/projects/Autism-detection-system

# Make script executable
chmod +x git-setup.sh

# Run the script
./git-setup.sh
```

The automated scripts will:
1. ✓ Initialize Git repository (if needed)
2. ✓ Set up remote repository
3. ✓ Add all files
4. ✓ Create initial commit
5. ✓ Push to GitHub

## Manual Setup

### Step 1: First Time Git Configuration

Configure your Git identity (only need to do this once):

```bash
# Set your name
git config --global user.name "Your Name"

# Set your email (use your GitHub email)
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

### Step 2: Initialize Git Repository

```bash
# Navigate to your project directory
cd "d:\projects\Capstone_Mine\Python-Autism Detection\Python-Autism Detection"

# Initialize Git (if not already done)
git init

# Check status
git status
```

### Step 3: Review .gitignore

Ensure your `.gitignore` file is properly configured to exclude:
- ✓ Virtual environments (`venv/`, `env/`)
- ✓ Python cache files (`__pycache__/`, `*.pyc`)
- ✓ Database files (`*.sqlite3`, `*.db`)
- ✓ Large model files (`*.h5`)
- ✓ Uploaded files (`static/uploads/`)
- ✓ IDE files (`.idea/`, `.vscode/`)
- ✓ Sensitive files (`.env`, `secrets.json`)

The `.gitignore` has already been configured for you!

## Creating GitHub Repository

### Option A: Using GitHub Website (Recommended for beginners)

1. **Go to GitHub**: https://github.com
2. **Click** the '+' icon in top right → "New repository"
3. **Fill in details**:
   - **Repository name**: `Autism-detection-system`
   - **Description**: "Machine learning-based autism detection system using CNN, questionnaires, and video analysis"
   - **Visibility**: Public (or Private if preferred)
   - **DO NOT** initialize with README, .gitignore, or license (we have these locally)
4. **Click** "Create repository"
5. **Copy** the repository URL (e.g., `https://github.com/yourusername/Autism-detection-system.git`)

### Option B: Using GitHub CLI

```bash
# Install GitHub CLI first: https://cli.github.com/

# Authenticate
gh auth login

# Create repository
gh repo create Autism-detection-system --public --source=. --remote=origin

# Push code
git push -u origin master
```

## Pushing to GitHub

### First Time Push

```bash
# Add remote repository (replace with your URL)
git remote add origin https://github.com/yourusername/Autism-detection-system.git

# Verify remote
git remote -v

# Check current branch
git branch

# If you're not on 'main' or 'master', rename it:
git branch -M main

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Autism Detection System with CNN, ML models, and Django web interface"

# Push to GitHub
git push -u origin main
```

### Subsequent Pushes

```bash
# Check status
git status

# Add changes
git add .

# Commit with meaningful message
git commit -m "Your descriptive commit message"

# Push to GitHub
git push
```

## Using the Provided Batch File (Windows)

You can also use the provided `git-push.bat` file:

```batch
# Edit git-push.bat to set your commit message
# Then run:
git-push.bat
```

## Troubleshooting

### Issue 1: "Permission denied (publickey)"

**Solution**: Set up SSH keys or use HTTPS with Personal Access Token

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
# Copy your public key:
cat ~/.ssh/id_ed25519.pub  # Linux/Mac
type %USERPROFILE%\.ssh\id_ed25519.pub  # Windows
```

### Issue 2: "Authentication failed"

**Solution**: Use Personal Access Token (PAT) instead of password

1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Select scopes: `repo`, `workflow`
4. Copy the token
5. Use token as password when pushing

```bash
# Cache credentials (Linux/Mac)
git config --global credential.helper cache

# Or use credential manager (Windows)
git config --global credential.helper wincred
```

### Issue 3: "Repository already exists"

**Solution**: Remove existing remote and add new one

```bash
# Remove existing remote
git remote remove origin

# Add new remote
git remote add origin YOUR_NEW_REPO_URL

# Push
git push -u origin main
```

### Issue 4: "Large files detected"

**Solution**: Use Git LFS for large files

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.h5"
git lfs track "*.pkl"

# Add .gitattributes
git add .gitattributes

# Commit and push
git commit -m "Add Git LFS tracking"
git push
```

### Issue 5: "Refusing to merge unrelated histories"

**Solution**: Allow unrelated histories

```bash
git pull origin main --allow-unrelated-histories
```

## Best Practices

### Commit Message Guidelines

✅ **Good commit messages**:
```
✓ Add CNN model training script with data augmentation
✓ Fix video upload path issue in views.py
✓ Update README with installation instructions
✓ Refactor database connection handling
```

❌ **Bad commit messages**:
```
✗ Update
✗ Fix bug
✗ Changes
✗ asdfgh
```

### Commit Frequency

- ✓ Commit often (after each logical change)
- ✓ One feature/fix per commit
- ✓ Test before committing
- ✓ Write descriptive messages

### Branch Strategy

For team projects, use branches:

```bash
# Create feature branch
git checkout -b feature/video-analysis

# Make changes and commit
git add .
git commit -m "Add real-time video analysis feature"

# Push branch
git push -u origin feature/video-analysis

# Create Pull Request on GitHub
# After merge, delete branch:
git branch -d feature/video-analysis
```

### What NOT to Commit

Never commit:
- ❌ Passwords or API keys
- ❌ Database files with personal data
- ❌ Large binary files (>100MB)
- ❌ Environment-specific configs
- ❌ Virtual environment folders
- ❌ Compiled files (`*.pyc`)

## Repository Settings (After First Push)

### 1. Add Repository Description

On GitHub repository page:
- Click "About" gear icon
- Add description: "Machine learning-based autism detection system using CNN, questionnaires, and video analysis"
- Add topics: `machine-learning`, `deep-learning`, `autism-detection`, `django`, `tensorflow`, `cnn`, `healthcare`, `python`

### 2. Enable GitHub Actions (Optional)

Create `.github/workflows/python-app.yml` for automated testing

### 3. Add Branch Protection

Settings → Branches → Add rule:
- Require pull request reviews
- Require status checks to pass

### 4. Set Up GitHub Pages (Optional)

For project documentation:
- Settings → Pages
- Source: Deploy from a branch
- Branch: `main` / `docs`

### 5. Add Collaborators

Settings → Collaborators → Add people

## Keeping Repository Updated

### Pulling Latest Changes

```bash
# Fetch and merge changes
git pull origin main

# Or fetch first, then merge
git fetch origin
git merge origin/main
```

### Sync Fork (If you forked)

```bash
# Add upstream remote
git remote add upstream https://github.com/original/repo.git

# Fetch upstream
git fetch upstream

# Merge upstream changes
git merge upstream/main
```

## GitHub Features to Explore

1. **Issues**: Track bugs and feature requests
2. **Projects**: Organize work with Kanban boards
3. **Wiki**: Create detailed documentation
4. **Insights**: View repository analytics
5. **Releases**: Tag stable versions
6. **Actions**: Automate CI/CD pipelines

## Quick Reference Commands

```bash
# Status and info
git status                  # Check current status
git log                     # View commit history
git diff                    # See changes
git branch                  # List branches

# Making changes
git add .                   # Stage all changes
git add file.py             # Stage specific file
git commit -m "message"     # Commit changes
git push                    # Push to remote

# Undoing changes
git restore file.py         # Discard changes
git reset --soft HEAD~1     # Undo last commit (keep changes)
git reset --hard HEAD~1     # Undo last commit (discard changes)

# Branching
git checkout -b new-branch  # Create and switch to branch
git checkout main           # Switch to main branch
git merge feature-branch    # Merge branch
git branch -d old-branch    # Delete branch

# Remote
git remote -v               # View remotes
git remote add origin URL   # Add remote
git pull                    # Pull changes
git push -u origin main     # Push and set upstream
```

## Getting Help

```bash
# Git help
git help
git help <command>

# Online resources
# - GitHub Docs: https://docs.github.com
# - Git Documentation: https://git-scm.com/doc
# - GitHub Learning Lab: https://lab.github.com
```

## Summary Checklist

Before pushing to GitHub, ensure:

- [ ] Git is installed and configured
- [ ] `.gitignore` is properly set up
- [ ] Sensitive data is removed/encrypted
- [ ] README.md is complete and informative
- [ ] requirements.txt is up to date
- [ ] Project structure is organized
- [ ] Code is tested and working
- [ ] Commit messages are descriptive
- [ ] GitHub repository is created
- [ ] Remote is added correctly
- [ ] All changes are committed
- [ ] Successfully pushed to GitHub

## Need More Help?

- 📧 Check GitHub documentation
- 💬 Ask on Stack Overflow
- 🐛 Open an issue in the repository
- 📚 Read Pro Git book (free): https://git-scm.com/book

---

**Happy Coding! 🚀**

*Last Updated: November 2025*


## Your Project is Now Git-Ready! 🎉

### Current Status
✅ Git repository initialized
✅ All files committed to local repository
✅ .gitignore configured properly
✅ Professional README.md created
✅ CONTRIBUTING.md added
✅ Project configured to use SQLite (no MySQL required)

---

## Step-by-Step: Push to GitHub

### 1. Create a New Repository on GitHub

1. Go to [GitHub.com](https://github.com)
2. Click the **"+"** icon in the top right
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name**: `autism-detection-system` (or your preferred name)
   - **Description**: "AI-powered autism detection system using Django and CNN"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

### 2. Connect Your Local Repository to GitHub

Copy and run these commands in your terminal:

```bash
# Navigate to your project directory
cd "d:\projects\Capstone_Mine\Python-Autism Detection - Copy\Python-Autism Detection"

# Add GitHub remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/autism-detection-system.git

# Verify the remote was added
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

**Replace** `YOUR_USERNAME` with your GitHub username!

### 3. Alternative: Using GitHub Desktop

If you prefer a GUI:

1. Download [GitHub Desktop](https://desktop.github.com/)
2. Install and sign in
3. Click **"Add"** → **"Add existing repository"**
4. Select your project folder
5. Click **"Publish repository"** button
6. Choose name, description, and visibility
7. Click **"Publish Repository"**

---

## After Pushing to GitHub

### Update README with Your Information

1. Edit `README.md`
2. Replace placeholders:
   - `yourusername` → your GitHub username
   - `your.email@example.com` → your email
   - Add screenshots if you have any

3. Commit and push changes:
```bash
git add README.md
git commit -m "docs: update README with personal information"
git push
```

### Add a License Badge (Optional)

If you want to add a license, go to your GitHub repo:
1. Click **"Add file"** → **"Create new file"**
2. Name it `LICENSE`
3. GitHub will offer a license template chooser
4. Select **MIT License** (recommended for open source)
5. Commit the file

---

## Recommended: Create a .env File for Secrets

For security, create a `.env` file (already in .gitignore):

```bash
# .env file
SECRET_KEY=your-new-secret-key-here
DEBUG=True
DATABASE_URL=sqlite:///db.sqlite3
```

Update `settings.py` to use environment variables:

```python
import os
from pathlib import Path

SECRET_KEY = os.environ.get('SECRET_KEY', 'vj&alc@@srub^mvv6wx7fwk@-j-+&kz268s&_*f-j1p43oi+xg')
DEBUG = os.environ.get('DEBUG', 'True') == 'True'
```

---

## Repository Structure Best Practices

### Create GitHub Actions for CI/CD (Advanced)

Create `.github/workflows/django.yml`:

```yaml
name: Django CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.10
    - name: Install Dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run Tests
      run: |
        python manage.py test
```

### Add Project Badges

Add these to your README.md (after creating the repo):

```markdown
![Build Status](https://github.com/yourusername/autism-detection/workflows/Django%20CI/badge.svg)
![GitHub stars](https://img.shields.io/github/stars/yourusername/autism-detection)
![GitHub forks](https://img.shields.io/github/forks/yourusername/autism-detection)
![GitHub issues](https://img.shields.io/github/issues/yourusername/autism-detection)
```

---

## Git Workflow for Future Updates

### Daily Workflow

```bash
# Check status
git status

# Add changes
git add .

# Commit with message
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug in video processing"
git commit -m "docs: update documentation"

# Push to GitHub
git push
```

### Commit Message Convention

Use semantic commit messages:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting)
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance tasks

### Creating Branches for Features

```bash
# Create and switch to a new branch
git checkout -b feature/new-model

# Make changes and commit
git add .
git commit -m "feat: implement improved CNN model"

# Push branch to GitHub
git push -u origin feature/new-model

# On GitHub, create a Pull Request to merge into main
```

---

## Protecting Your Repository

### Things to NEVER commit:

- ✅ Already in `.gitignore`:
  - `db.sqlite3` (database with user data)
  - `*.pyc` (compiled Python files)
  - `venv/` (virtual environment)
  - `static/uploads/*` (user uploads)
  - `*.h5` (large model files)
  - `.env` (environment variables)

### If You Accidentally Commit Secrets:

```bash
# Remove file from Git but keep locally
git rm --cached sensitive_file.py

# Add to .gitignore
echo "sensitive_file.py" >> .gitignore

# Commit the removal
git add .gitignore
git commit -m "chore: remove sensitive file from git"
git push

# For exposed secrets, regenerate them immediately!
```

---

## Sharing Your Project

### Make Your Project Stand Out

1. **Add Screenshots** to README
2. **Create a demo video** (upload to YouTube)
3. **Write a blog post** about the project
4. **Add to your portfolio** website
5. **Share on LinkedIn**

### Example README Sections to Add

```markdown
## 📸 Screenshots

![Home Page](screenshots/home.png)
![Analysis Page](screenshots/analysis.png)

## 🎥 Demo Video

[Watch Demo on YouTube](https://youtube.com/your-video)

## 🏆 Achievements

- 95% accuracy in autism detection
- Processes 30 frames per second
- Used by 100+ testers
```

---

## Quick Command Reference

```bash
# Check repository status
git status

# View commit history
git log --oneline

# See what changed
git diff

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard all local changes
git restore .

# Pull latest changes from GitHub
git pull

# Clone your repo elsewhere
git clone https://github.com/yourusername/autism-detection.git
```

---

## Need Help?

- [GitHub Docs](https://docs.github.com)
- [Git Documentation](https://git-scm.com/doc)
- [GitHub Education](https://education.github.com)

---

## Next Steps

1. ✅ Push to GitHub (see Section 2)
2. ⬜ Add screenshots
3. ⬜ Update README with your info
4. ⬜ Add project to your resume/portfolio
5. ⬜ Share with the community!

Good luck with your project! 🚀

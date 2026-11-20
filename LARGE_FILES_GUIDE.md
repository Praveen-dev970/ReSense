# 📦 Handling Large Files in Autism Detection System

## Overview

This project contains large files that cannot be pushed directly to GitHub:
- **Model files** (~100MB+): `autism_detection_model.h5`
- **Video files** (50MB+): `*.mp4`, `*.avi`
- **Dataset images** (100MB+): Training and test set images
- **Duplicate folder**: `Python-Autism Detection/` subfolder

GitHub has a file size limit of **100MB** per file and repositories should generally stay under **1GB**.

## Solutions

### Option 1: Use Git Large File Storage (LFS) - Recommended

Git LFS is designed for handling large files in Git repositories.

#### Setup Git LFS

```bash
# Install Git LFS (one-time setup)
# Windows: Download from https://git-lfs.github.com/
# Mac: brew install git-lfs
# Linux: sudo apt-get install git-lfs

# Initialize Git LFS
git lfs install

# Track large file types
git lfs track "*.h5"
git lfs track "*.mp4"
git lfs track "*.avi"
git lfs track "dataset/**"

# Add .gitattributes
git add .gitattributes

# Commit and push
git add .
git commit -m "Add Git LFS for large files"
git push origin master
```

#### Git LFS Limitations
- Free tier: 1GB storage, 1GB bandwidth/month
- Paid plans available for more storage

### Option 2: Use External Storage (Recommended for Very Large Datasets)

Store large files externally and provide download links.

#### A. Google Drive / Dropbox

1. Upload files to cloud storage
2. Get shareable link
3. Add download instructions to README

```markdown
## Downloading Large Files

Due to GitHub size limitations, large files are hosted externally:

- **Model File** (`autism_detection_model.h5`): 
  [Download from Google Drive](YOUR_LINK_HERE)
  
- **Training Dataset**: 
  [Download from Google Drive](YOUR_LINK_HERE)
  
- **Test Dataset**: 
  [Download from Google Drive](YOUR_LINK_HERE)

Place downloaded files in the following structure:
```
project/
├── autism_detection_model.h5
├── dataset/
│   ├── training_set/
│   └── test_set/
```
```

#### B. GitHub Releases

For model files, use GitHub Releases:

1. Go to your repository on GitHub
2. Click "Releases" → "Create a new release"
3. Tag: `v1.0` or `models-v1`
4. Upload your `.h5` model file
5. Publish release

Link in README:
```markdown
Download the trained model from [Releases](https://github.com/Venu22003/Autism-detection-system/releases)
```

#### C. Kaggle / Hugging Face

For datasets:
- **Kaggle**: Upload dataset as Kaggle Dataset
- **Hugging Face**: Upload to Hugging Face Hub

### Option 3: Exclude from Repository (Quick Fix)

Simply don't include large files in the repository.

```bash
# Remove large files from tracking (keep local copies)
git rm --cached autism_detection_model.h5
git rm --cached *.mp4
git rm --cached -r dataset/training_set/
git rm --cached -r dataset/test_set/
git rm --cached -r "Python-Autism Detection/"

# Commit changes
git commit -m "Remove large files from repository"

# Push
git push origin master
```

Then add to README:
```markdown
## Setup Instructions

### Download Required Files

This repository does not include the following large files:
- Trained model: `autism_detection_model.h5` (Download link or train yourself)
- Training images: Place in `dataset/training_set/`
- Test images: Place in `dataset/test_set/`

### Training Your Own Model

If you don't have the pre-trained model, you can train it:

```bash
python cnn.py
```

This will create `autism_detection_model.h5` in the project root.
```

### Option 4: Use .gitignore (Prevention)

The `.gitignore` has been updated to exclude large files going forward.

Files now ignored:
```
*.h5              # Model files
*.mp4, *.avi      # Video files
dataset/          # Image datasets
*.sql             # Database dumps
Python-Autism Detection/  # Duplicate folder
```

## Recommended Strategy for This Project

### For Model Files (`*.h5`)
**Use GitHub Releases** - Upload trained model as a release asset

### For Datasets
**Use External Link** - Provide download link in README with setup instructions

### For Videos
**Provide Sample** - Include 1-2 small sample videos, link to full dataset

### For Nested Folder
**Remove** - Delete the duplicate `Python-Autism Detection/` folder

## Step-by-Step: Clean Repository and Push

```bash
# 1. Navigate to project
cd "d:\projects\Capstone_Mine\Python-Autism Detection\Python-Autism Detection"

# 2. Remove large files from Git (keeps local files)
git rm --cached -r "Python-Autism Detection/" 2>$null
git rm --cached autism_detection_model.h5 2>$null
git rm --cached "*.mp4" 2>$null
git rm --cached -r dataset/training_set/ 2>$null
git rm --cached -r dataset/test_set/ 2>$null
git rm --cached database/autism.sql 2>$null

# 3. Commit changes
git add .gitignore
git commit -m "Remove large files and update .gitignore"

# 4. Push to GitHub
git push origin master

# 5. Create GitHub Release for model
# - Go to GitHub repository
# - Releases → New Release
# - Tag: v1.0
# - Upload autism_detection_model.h5
# - Publish
```

## After Cleaning Repository

Update README.md with download instructions:

```markdown
## Setup Instructions

### 1. Clone Repository
\`\`\`bash
git clone https://github.com/Venu22003/Autism-detection-system.git
cd Autism-detection-system
\`\`\`

### 2. Download Large Files

#### Option A: Download Pre-trained Model
Download `autism_detection_model.h5` from [Releases](https://github.com/Venu22003/Autism-detection-system/releases)

Place in project root:
\`\`\`
Autism-detection-system/
├── autism_detection_model.h5  ← Place here
├── ...
\`\`\`

#### Option B: Train Your Own Model
\`\`\`bash
# Prepare your dataset in dataset/ folder
python cnn.py
\`\`\`

### 3. Prepare Dataset

Structure your dataset as follows:
\`\`\`
dataset/
├── training_set/
│   ├── Autistic/
│   │   └── (images)
│   └── Non_Autistic/
│       └── (images)
└── test_set/
    ├── Autistic/
    │   └── (images)
    └── Non_Autistic/
        └── (images)
\`\`\`

### 4. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 5. Run Application
\`\`\`bash
python manage.py runserver
\`\`\`
```

## Checking Repository Size

```bash
# Check size before push
git count-objects -vH

# List large files
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sed -n 's/^blob //p' | sort --numeric-sort --key=2 | tail -n 10
```

## Alternative: Start Fresh

If the repository is too messy:

```bash
# 1. Backup important files
mkdir ../backup
cp -r . ../backup/

# 2. Remove git history
rm -rf .git

# 3. Initialize new repository
git init

# 4. Add only essential files
git add .
git commit -m "Initial commit - Clean repository"

# 5. Force push to GitHub
git remote add origin https://github.com/Venu22003/Autism-detection-system.git
git push -u --force origin master
```

## Summary

| File Type | Size | Solution |
|-----------|------|----------|
| Model files (`.h5`) | 100MB+ | GitHub Releases |
| Training images | 200MB+ | External link or train instructions |
| Test images | 50MB+ | External link |
| Video files (`.mp4`) | 50MB+ | Sample only, link to full dataset |
| Code files | <1MB | Include in repository |

---

**Need Help?**
- Git LFS Docs: https://git-lfs.github.com/
- GitHub File Size Limits: https://docs.github.com/en/repositories/working-with-files/managing-large-files

*Last Updated: November 2025*

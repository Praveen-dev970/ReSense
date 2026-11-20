# Quick Start Guide - Autism Detection Project

## Option 1: Automated Setup (Recommended)

### Using PowerShell Script
```powershell
# Run the automated setup script
.\setup.ps1

# Then follow the manual steps below for database setup
```

## Option 2: Manual Setup

### Step 1: Install Python Dependencies
```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### Step 2: Setup MySQL Database

**Important:** Make sure MySQL is installed and running!

1. Open MySQL command line or MySQL Workbench
2. Run these commands:

```sql
CREATE DATABASE autism;
USE autism;

CREATE TABLE Users (
    name VARCHAR(100),
    mailid VARCHAR(100),
    mobileno VARCHAR(20),
    userid VARCHAR(50) PRIMARY KEY,
    password VARCHAR(50)
);

CREATE TABLE vfiles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    vpath VARCHAR(255),
    vdescr TEXT
);
```

### Step 3: Update Database Password (if needed)

Edit `textutils/settings.py` and update the password:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'autism',
        'USER': 'root',
        'PASSWORD': '',  # Add your MySQL password here
        'HOST': '127.0.0.1',
        'PORT': '3306',
    }
}
```

### Step 4: Run Django Setup
```powershell
# Make sure virtual environment is activated
python manage.py migrate
```

### Step 5: Start the Server
```powershell
python manage.py runserver
```

Or simply run:
```powershell
.\run.bat
```

## Access the Application

- **Main Page:** http://127.0.0.1:8000/
- **Registration:** http://127.0.0.1:8000/registration/
- **User Login:** http://127.0.0.1:8000/login/
- **Admin Login:** http://127.0.0.1:8000/login1
  - Username: `admin`
  - Password: `admin`

## Common Issues & Solutions

### 1. MySQL Connection Error
**Error:** `Can't connect to MySQL server`

**Solution:**
- Ensure MySQL service is running
- Check Windows Services for MySQL
- Verify credentials in `settings.py`

### 2. mysqlclient Installation Fails
**Error:** `error: Microsoft Visual C++ 14.0 is required`

**Solution A:** Install pymysql instead
```powershell
pip install pymysql
```
Then uncomment these lines in `textutils/__init__.py`:
```python
import pymysql
pymysql.install_as_MySQLdb()
```

**Solution B:** Download and install mysqlclient wheel
1. Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#mysqlclient
2. Install: `pip install mysqlclient‑1.4.6‑cp310‑cp310‑win_amd64.whl`

### 3. TensorFlow Installation Issues
**Error:** TensorFlow version not compatible

**Solution:**
```powershell
pip uninstall tensorflow
pip install tensorflow==2.10.0
```

### 4. Port Already in Use
**Error:** `Error: That port is already in use`

**Solution:**
```powershell
python manage.py runserver 8080
```

### 5. Static Files Not Loading
**Solution:**
```powershell
python manage.py collectstatic
```

### 6. Dataset Images Missing
**Solution:**
- Ensure training images are in `dataset/training_set/Autistic/` and `dataset/training_set/Non_Autistic/`
- Ensure test images are in `dataset/test_set/Autistic/` and `dataset/test_set/Non_Autistic/`
- You need actual image files for the CNN model to work

## Project Features

1. **User Registration & Login** - Create account and login
2. **Questionnaire Analysis** - Answer autism screening questions
3. **Video Analysis** - Upload child behavior videos for AI analysis
4. **Admin Panel** - View uploaded videos and results

## File Structure
```
├── manage.py              # Django entry point
├── requirements.txt       # Python dependencies
├── setup.ps1             # Automated setup script
├── run.bat               # Quick start script
├── textutils/            # Django app
│   ├── settings.py       # Configuration
│   ├── urls.py           # URL routing
│   └── views.py          # Business logic
├── templates/            # HTML files
├── static/uploads/       # Uploaded videos
├── dataset/              # Training images
│   ├── training_set/
│   │   ├── Autistic/
│   │   └── Non_Autistic/
│   └── test_set/
│       ├── Autistic/
│       └── Non_Autistic/
└── database/             # SQL schemas
```

## Testing the Application

1. Register a new user
2. Login with credentials
3. Try the questionnaire at `/analyze`
4. Upload a video at `/find`

## Need Help?

Check the full documentation in `SETUP_INSTRUCTIONS.md`

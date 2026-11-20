# Autism Detection Project - Setup Instructions

## Prerequisites
- Python 3.7 or higher (Python 3.10 detected)
- MySQL Server installed and running
- Virtual environment (recommended)

## Step-by-Step Setup

### 1. Create and Activate Virtual Environment (Recommended)
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Note:** If `mysqlclient` installation fails on Windows, try:
```powershell
# Alternative: Install pymysql instead
pip install pymysql
```
Then add this to `textutils/__init__.py`:
```python
import pymysql
pymysql.install_as_MySQLdb()
```

### 3. Setup MySQL Database
1. Start your MySQL server
2. Create the database:
```sql
CREATE DATABASE autism;
USE autism;
```
3. Import the database schema:
```sql
source database/autism.sql
```
Or manually create tables:
```sql
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

### 4. Configure Database Settings
Update `textutils/settings.py` if your MySQL credentials are different:
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'autism',
        'USER': 'root',
        'PASSWORD': 'your_password',  # Update if needed
        'HOST': '127.0.0.1',
        'PORT': '3306',
    }
}
```

### 5. Setup Static Files and Uploads Directory
```powershell
# Make sure these directories exist
mkdir static\uploads -Force
mkdir output_frames -Force
mkdir dataset\training_set\Autistic -Force
mkdir dataset\training_set\Non_Autistic -Force
mkdir dataset\test_set\Autistic -Force
mkdir dataset\test_set\Non_Autistic -Force
```

### 6. Run Django Migrations
```powershell
python manage.py makemigrations
python manage.py migrate
```

### 7. Create Superuser (Optional)
```powershell
python manage.py createsuperuser
```

### 8. Run the Development Server
```powershell
python manage.py runserver
```

The application will be available at: `http://127.0.0.1:8000/`

## Application Features

### 1. User Registration and Login
- Navigate to `/registration/` to create a new account
- Login at `/login/` with your credentials

### 2. Admin Access
- Login at `/login1` with credentials: `admin` / `admin`

### 3. Autism Detection
- Answer questionnaire at `/analyze`
- Upload video for behavioral analysis at `/find`

## Troubleshooting

### MySQL Connection Issues
- Ensure MySQL server is running
- Check credentials in `settings.py`
- Verify the `autism` database exists

### TensorFlow Issues
If TensorFlow 2.6.2 fails to install:
```powershell
pip install tensorflow==2.10.0
```

### OpenCV Issues
If opencv-python fails:
```powershell
pip install opencv-python-headless
```

### Port Already in Use
```powershell
python manage.py runserver 8080
```

## Project Structure
```
├── manage.py              # Django management script
├── main.py               # Video processing script
├── cnn.py                # CNN model training
├── predict.py            # Prediction script
├── textutils/            # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── views.py
├── templates/            # HTML templates
├── static/               # Static files
│   └── uploads/          # Uploaded videos
├── dataset/              # Training/test images
├── database/             # SQL files
└── output_frames/        # Processed frames

```

## Admin Credentials
- Username: `admin`
- Password: `admin`

## Notes
- Ensure dataset images are properly organized before training
- The model file `autism_detection_model.h5` should be in the root directory
- Video files must be uploaded to `static/uploads/` for processing

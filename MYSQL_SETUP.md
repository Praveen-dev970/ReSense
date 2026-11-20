# MySQL Setup and Start Guide for Windows

## Current Status
✗ MySQL server is not running on your system

## Solution 1: Start MySQL Service (If Already Installed)

### Option A: Using Windows Services GUI
1. Press `Win + R` to open Run dialog
2. Type `services.msc` and press Enter
3. Look for "MySQL" or "MySQL80" service
4. Right-click on it and select "Start"
5. Set it to "Automatic" startup if you want it to start on boot

### Option B: Using Command Line (Run as Administrator)
```powershell
# Start MySQL service
net start MySQL

# Or if you have MySQL80
net start MySQL80

# Check service status
sc query MySQL
```

### Option C: Using PowerShell (Run as Administrator)
```powershell
# Start MySQL service
Start-Service MySQL

# Or
Start-Service MySQL80

# Check if running
Get-Service MySQL*
```

## Solution 2: Install MySQL (If Not Installed)

### Option A: MySQL Installer (Recommended)
1. Download MySQL Installer from: https://dev.mysql.com/downloads/installer/
2. Choose "mysql-installer-community" (Windows x86, 32-bit)
3. Run the installer
4. Select "Developer Default" or "Server only"
5. During installation:
   - Set root password (or leave blank)
   - Configure to run as Windows Service
   - Set to start automatically

### Option B: XAMPP (Easiest for Development)
1. Download XAMPP from: https://www.apachefriends.org/
2. Install XAMPP
3. Open XAMPP Control Panel
4. Click "Start" button next to MySQL
5. Default credentials: root / (no password)

### Option C: MySQL Workbench + Server
1. Download from: https://dev.mysql.com/downloads/mysql/
2. Install MySQL Community Server
3. Install MySQL Workbench (GUI tool)
4. Configure and start the server

## Solution 3: Use SQLite Instead (No Installation Required)

### Quick Switch to SQLite
1. Edit `textutils/settings.py`
2. Replace the DATABASES section with:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```

3. Run migrations:
```powershell
python manage.py migrate
```

4. SQLite will work immediately without any server!

**Note:** If using SQLite, you'll need to modify some raw SQL queries in `views.py` 
to use Django ORM instead, but the project will run.

## After MySQL is Running

### Step 1: Test the connection
```powershell
python test_mysql.py
```

### Step 2: Run Django migrations
```powershell
python manage.py migrate
```

### Step 3: Start the development server
```powershell
python manage.py runserver
```

## Troubleshooting

### Can't find MySQL service
- MySQL might not be installed
- Check in Services: Look for any service with "MySQL" in the name
- Check XAMPP if you installed it

### Access Denied Error
- Update the password in `textutils/settings.py`
- Check MySQL root password

### Port 3306 Already in Use
- Another application is using MySQL's port
- Check with: `netstat -ano | findstr :3306`
- Change port in MySQL configuration

## Quick Test Commands

### Check if MySQL port is open:
```powershell
Test-NetConnection -ComputerName localhost -Port 3306
```

### Check if MySQL process is running:
```powershell
Get-Process | Where-Object {$_.ProcessName -like "*mysql*"}
```

### Alternative: Use MySQL Command Line
```powershell
# If MySQL is installed, try:
mysql -u root -p

# Then create the database:
CREATE DATABASE autism;
```

## Recommendation for Quick Start

**For development purposes, I recommend using SQLite** (Solution 3) as it:
- Requires no installation
- No server to manage
- Works immediately
- Perfect for testing and development
- Can switch to MySQL later for production

Would you like me to help you switch to SQLite so you can start the project right away?

"""
Test MySQL Database Connection
This script tests if MySQL is accessible and the autism database exists
"""
import pymysql
import sys

def test_mysql_connection():
    print("="*60)
    print("Testing MySQL Database Connection")
    print("="*60)
    
    try:
        # Try to connect to MySQL
        print("\n1. Attempting to connect to MySQL server...")
        connection = pymysql.connect(
            host='127.0.0.1',
            user='root',
            password='',  # Change if you have a password
            port=3306
        )
        print("   ✓ Successfully connected to MySQL server!")
        
        # Check if autism database exists
        print("\n2. Checking if 'autism' database exists...")
        cursor = connection.cursor()
        cursor.execute("SHOW DATABASES LIKE 'autism'")
        result = cursor.fetchone()
        
        if result:
            print("   ✓ Database 'autism' exists!")
            
            # Connect to autism database
            connection.select_db('autism')
            print("\n3. Checking tables in 'autism' database...")
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            if tables:
                print("   ✓ Found the following tables:")
                for table in tables:
                    print(f"      - {table[0]}")
            else:
                print("   ⚠ Warning: No tables found in 'autism' database")
                print("   You need to import the database schema!")
                return False
        else:
            print("   ✗ Database 'autism' does not exist!")
            print("\n   Creating 'autism' database...")
            cursor.execute("CREATE DATABASE autism")
            print("   ✓ Database 'autism' created successfully!")
            
            print("\n   Creating required tables...")
            connection.select_db('autism')
            
            # Create Users table
            cursor.execute("""
                CREATE TABLE Users (
                    name VARCHAR(100),
                    mailid VARCHAR(100),
                    mobileno VARCHAR(20),
                    userid VARCHAR(50) PRIMARY KEY,
                    password VARCHAR(50)
                )
            """)
            print("   ✓ Created 'Users' table")
            
            # Create vfiles table
            cursor.execute("""
                CREATE TABLE vfiles (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    vpath VARCHAR(255),
                    vdescr TEXT
                )
            """)
            print("   ✓ Created 'vfiles' table")
            
            connection.commit()
        
        cursor.close()
        connection.close()
        
        print("\n" + "="*60)
        print("✓ MySQL setup is complete and ready!")
        print("="*60)
        return True
        
    except pymysql.Error as e:
        print(f"\n✗ MySQL Error: {e}")
        print("\nPossible solutions:")
        print("1. Make sure MySQL server is installed and running")
        print("2. Check if the MySQL service is started in Windows Services")
        print("3. Verify your MySQL credentials (username/password)")
        print("4. Make sure MySQL is listening on port 3306")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected Error: {e}")
        return False

if __name__ == "__main__":
    success = test_mysql_connection()
    sys.exit(0 if success else 1)

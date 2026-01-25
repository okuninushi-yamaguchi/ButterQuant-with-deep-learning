import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from database import DatabaseManager

def test_conn():
    # Force use_pg for testing
    os.environ['DB_HOST'] = 'localhost'
    os.environ['DB_NAME'] = 'butterquant'
    os.environ['DB_USER'] = 'postgres'
    os.environ['DB_PASSWORD'] = 'butterquant_pass'
    
    db = DatabaseManager()
    print(f"PostgreSQL enabled: {db.use_pg}")
    
    try:
        with db.get_connection() as conn:
            print("Successfully connected to PostgreSQL!")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM analysis_history")
            count = cursor.fetchone()[0]
            print(f"Rows in analysis_history: {count}")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_conn()

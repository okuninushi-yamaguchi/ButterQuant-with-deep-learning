import sqlite3
import psycopg2
from psycopg2 import extras
import re
import os
import json

# Database Configuration
SQLITE_DB_MARKET = "backend/data/market_research.db"
SQLITE_DB_HISTORY = "backend/data/history.db"

PG_CONN_PARAMS = {
    "dbname": "butterquant",
    "user": "postgres",
    "password": "butterquant_pass",
    "host": "localhost",
    "port": 5432
}

def clean_json(json_str):
    if not json_str:
        return None
    # Replace unquoted NaN, Infinity, -Infinity with null (case-insensitive)
    json_str = re.sub(r'\bNaN\b', 'null', json_str, flags=re.IGNORECASE)
    json_str = re.sub(r'\bInfinity\b', 'null', json_str, flags=re.IGNORECASE)
    json_str = re.sub(r'-Infinity\b', 'null', json_str, flags=re.IGNORECASE)
    return json_str

def migrate_daily_metrics():
    print("Starting migration of daily_metrics...")
    sqlite_conn = sqlite3.connect(SQLITE_DB_MARKET)
    sqlite_cursor = sqlite_conn.cursor()
    
    pg_conn = psycopg2.connect(**PG_CONN_PARAMS)
    pg_cursor = pg_conn.cursor()
    
    sqlite_cursor.execute("SELECT * FROM daily_metrics")
    rows = sqlite_cursor.fetchall()
    
    # Get column names
    sqlite_cursor.execute("PRAGMA table_info(daily_metrics)")
    columns = [col[1] for col in sqlite_cursor.fetchall()]
    
    insert_query = f"INSERT INTO daily_metrics ({', '.join(columns)}) VALUES %s ON CONFLICT DO NOTHING"
    
    extras.execute_values(pg_cursor, insert_query, rows)
    
    pg_conn.commit()
    print(f"Migrated {len(rows)} rows to daily_metrics.")
    
    sqlite_conn.close()
    pg_conn.close()

def migrate_analysis_history():
    print("Starting migration of analysis_history...")
    sqlite_conn = sqlite3.connect(SQLITE_DB_HISTORY)
    sqlite_cursor = sqlite_conn.cursor()
    
    pg_conn = psycopg2.connect(**PG_CONN_PARAMS)
    pg_cursor = pg_conn.cursor()
    
    sqlite_cursor.execute("SELECT ticker, analysis_date, total_score, butterfly_type, recommendation, full_result, created_at FROM analysis_history")
    rows = sqlite_cursor.fetchall()
    print(f"Found {len(rows)} rows in SQLite analysis_history.")
    
    processed_rows = []
    for row in rows:
        row_list = list(row)
        # Clean the JSON field (full_result is at index 5)
        row_list[5] = clean_json(row_list[5])
        processed_rows.append(tuple(row_list))
    
    insert_query = """
    INSERT INTO analysis_history (ticker, analysis_date, total_score, butterfly_type, recommendation, full_result, created_at)
    VALUES %s
    """
    
    extras.execute_values(pg_cursor, insert_query, processed_rows)
    
    pg_conn.commit()
    print(f"Migrated {len(processed_rows)} rows to analysis_history.")
    
    sqlite_conn.close()
    pg_conn.close()

if __name__ == "__main__":
    try:
        migrate_daily_metrics()
        migrate_analysis_history()
        print("Migration completed successfully!")
    except Exception as e:
        print(f"Migration failed: {e}")

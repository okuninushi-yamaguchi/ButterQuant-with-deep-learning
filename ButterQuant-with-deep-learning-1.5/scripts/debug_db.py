import sqlite3
import os

db_path = 'backend/data/history.db'
if not os.path.exists(db_path):
    print("DB not found")
else:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("--- Unique Tickers in LAST 24 HOURS ---")
        cursor.execute("""
            SELECT ticker, COUNT(*) 
            FROM analysis_history 
            WHERE created_at >= datetime('now', '-1 day')
            GROUP BY ticker 
            ORDER BY COUNT(*) DESC
        """)
        rows = cursor.fetchall()
        for r in rows:
            print(f"{r[0]}: {r[1]}")
            
        print(f"\nTotal Unique Recent Tickers: {len(rows)}")
        
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

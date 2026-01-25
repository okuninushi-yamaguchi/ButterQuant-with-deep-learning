import sqlite3

def get_schema(db_path, out_file):
    out_file.write(f"--- Schema for {db_path} ---\n")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table in tables:
            table_name = table[0]
            out_file.write(f"\nTable: {table_name}\n")
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            for col in columns:
                out_file.write(f"  {col[1]} ({col[2]})\n")
        conn.close()
    except Exception as e:
        out_file.write(f"Error: {e}\n")

if __name__ == "__main__":
    with open("schema_output.txt", "w") as f:
        get_schema("backend/data/market_research.db", f)
        get_schema("backend/data/history.db", f)

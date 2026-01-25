import sqlite3
import json

import os

try:
    # 动态获取相关路径 / Dynamic path resolution
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'backend', 'data', 'history.db')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT full_result FROM analysis_history WHERE ticker='AZO' ORDER BY id DESC LIMIT 1")
    res = cursor.fetchone()
    if res:
        data = json.loads(res[0])
        print(f"Ticker: AZO")
        print(f"Type from butterfly: {data['butterfly'].get('butterfly_type')}")
        print(f"Type from fourier: {data['fourier'].get('butterfly_type')}")
    else:
        print("No data found for AZO")
except Exception as e:
    print(f"Error: {e}")

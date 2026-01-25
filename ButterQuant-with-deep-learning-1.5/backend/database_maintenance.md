# Database Maintenance Guide

ButterQuant uses SQLite as the local database to store historical analysis results from daily batch scanning.

## 📁 Database Locations
- **Main Database**: `backend/data/history.db`
- **Research Database**: `backend/data/market_research.db`
- **Type**: SQLite 3

## 📊 Database Schema

### `analysis_history` (Main Table)
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INTEGER PRIMARY KEY | Auto-incrementing primary key |
| `ticker` | TEXT | Stock ticker (e.g., AAPL) |
| `analysis_date` | TEXT | Analysis timestamp (ISO 8601 format) |
| `total_score` | REAL | Quant strategy total score (0-100) |
| `butterfly_type` | TEXT | Strategy type (CALL/PUT/IRON) |
| `recommendation` | TEXT | Action (STRONG_BUY/BUY/NEUTRAL/AVOID) |
| `full_result` | TEXT (JSON) | Complete analysis result as a JSON string |
| `created_at` | TIMESTAMP | Record creation timestamp |

### `daily_metrics` (Research Table)
Stores flattened daily analysis data optimized for SQL queries and AI training. Ideal for batch analysis via Pandas.

**Key Fields**:
- `ticker`, `analysis_date`, `current_price`
- `trend_direction`, `trend_slope` (Fourier Analysis)
- `predicted_vol`, `vol_mispricing` (GARCH Volatility)
- `delta`, `gamma`, `vega` (Option Greeks)
- `profit_ratio`, `prob_profit` (Strategy Metrics)

**Pandas Query Example**:
```python
import sqlite3
import pandas as pd
conn = sqlite3.connect('backend/data/market_research.db')
df = pd.read_sql("SELECT * FROM daily_metrics WHERE total_score > 80", conn)
```

## 🛠️ Common Operations

### 1. Database Backup
It is recommended to back up database files regularly.
```bash
# Windows
copy backend\data\history.db backend\data\history_backup.db
```

### 2. View Data Statistics (Python)
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('backend/data/history.db')

# View total record count
count = pd.read_sql_query("SELECT COUNT(*) FROM analysis_history", conn)
print(f"Total records: {count.iloc[0,0]}")

# View recent high-score opportunities
top_picks = pd.read_sql_query("""
    SELECT ticker, total_score, butterfly_type, analysis_date 
    FROM analysis_history 
    WHERE total_score > 80 
    ORDER BY analysis_date DESC LIMIT 10
""", conn)
print(top_picks)

conn.close()
```

### 3. Data Cleanup
If the database file becomes too large, you can prune old historical records (e.g., keep the last 60 days).
```sql
DELETE FROM analysis_history 
WHERE analysis_date < date('now', '-60 days');

VACUUM; -- Reclaim unused space and reduce file size
```

### 4. Integrity Check
If you encounter database corruption errors, run:
```sql
PRAGMA integrity_check;
```

## 🔄 Migration & Upgrades
If you need to add fields or migrate to PostgreSQL in the future:
1. Use the `init_db` function in `backend/database.py` to modify the schema.
2. Write a migration script to export from SQLite and import into the new environment.

## ✍️ Development Note
Database interaction logic is encapsulated in the `DatabaseManager` class within `backend/database.py`.
- `save_analysis(result)`: Saves a single analysis result to the main DB.
- `save_daily_metrics(ticker, result)`: Saves flattened metrics to the research DB.

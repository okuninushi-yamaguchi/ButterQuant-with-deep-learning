import os
import sqlite3
from backend.ticker_utils import get_tickers_with_tags

# Imitate DatabaseManager init
base_dir = os.path.abspath('backend')
project_root = os.path.dirname(base_dir) 
doc_dir = os.path.join(project_root, 'doc')

file_map = {
    'NASDAQ': os.path.join(doc_dir, 'nas100.md'),
    'SP500': os.path.join(doc_dir, 'sp500.md')
}

print(f"Loading tags from: {file_map}")
try:
    ticker_tags = get_tickers_with_tags(file_map)
    print(f"Loaded tags for {len(ticker_tags)} tickers")
    print(f"Sample tags (AAPL): {ticker_tags.get('AAPL')}")
    print(f"Sample tags (NVDA): {ticker_tags.get('NVDA')}")
    print(f"Sample tags (MSTR): {ticker_tags.get('MSTR')}")
except Exception as e:
    print(f"Error loading tags: {e}")

# Check DB tickers
db_path = 'backend/data/history.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("SELECT DISTINCT ticker FROM analysis_history")
db_tickers = [row[0] for row in cursor.fetchall()]
conn.close()

print(f"DB has {len(db_tickers)} unique tickers")

# Check intersection
count_nasdaq = 0
found_nasdaq = []
for t in db_tickers:
    tags = ticker_tags.get(t, [])
    if 'NASDAQ' in tags:
        count_nasdaq += 1
        if len(found_nasdaq) < 10:
            found_nasdaq.append(t)

print(f"DB tickers with NASDAQ tag: {count_nasdaq}")
print(f"First 10 NASDAQ found: {found_nasdaq}")

# Why missing?
print("\nChecking matching for missing ones:")
for target in ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'META', 'GOOGL']:
    if target in db_tickers:
        print(f"{target} in DB: YES. Tags: {ticker_tags.get(target)}")
    else:
        print(f"{target} in DB: NO")

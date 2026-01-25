import pandas as pd

def analyze():
    try:
        df = pd.read_parquet('ml/training_data.parquet')
        print(f"--- Dataset Statistics ---")
        print(f"Total Rows: {len(df)}")
        print(f"Success Count: {df['success'].sum()} ({df['success'].mean():.2%})")
        print(f"Average ROI: {df['roi_est'].mean():.2f}")
        print(f"Unique Tickers: {df['ticker'].nunique()}")
        
        # Check for NaNs in features
        feature_cols = ['trend_slope', 'dominant_period', 'predicted_vol', 'iv_percentile', 'delta', 'gamma', 'vega', 'theta']
        for col in feature_cols:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    print(f"Feature '{col}' has {nan_count} NaNs")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze()

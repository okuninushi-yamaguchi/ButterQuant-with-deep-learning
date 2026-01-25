"""
Fix Label Bias Script / 修复标签偏差脚本

功能 / Functionality:
1. Load existing training data (parquet) / 加载现有训练数据
2. Recalculate ROI using dynamic DTE timing / 使用动态DTE时间重新计算ROI
3. Update labels based on new thresholds / 基于新阈值更新标签
4. Support Test Mode (--limit) / 支持测试模式

Usage:
    python ml/fix_label_bias.py --limit 5  (Test Mode)
    python ml/fix_label_bias.py --apply    (Apply changes to full dataset)
"""

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import logging
from datetime import timedelta
import sys
import os

# Ensure backend modules can be imported if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml.features import calculate_dynamic_evaluation_date, classify_roi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress yfinance noise
logging.getLogger('yfinance').setLevel(logging.CRITICAL)


def calculate_butterfly_roi(future_price, lower, center, upper, cost, max_profit):
    """计算蝴蝶策略ROI / Calculate Butterfly ROI"""
    if lower <= future_price <= upper:
        if future_price <= center:
            payoff = max_profit * (future_price - lower) / (center - lower) if center != lower else 0
        else:
            payoff = max_profit * (upper - future_price) / (upper - center) if upper != center else 0
    else:
        payoff = -cost
    
    roi = (payoff - cost) / cost if cost > 0 else -1.0
    return roi


def process_dataset(input_path, output_path, limit=None, apply_changes=False):
    """
    处理数据集 / Process Dataset
    """
    if not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    # If in test mode, limit sample size
    if limit:
        logger.info(f"Test Mode: Processing only first {limit} samples.")
        df = df.head(limit).copy()
    else:
        logger.info(f"Processing full dataset: {len(df)} samples.")

    # Results container
    results = []
    
    successful = 0
    failed = 0

    print(f"\n{'Ticker':<8} | {'DTE':<4} | {'Eval Date (Old/New)':<25} | {'ROI (Old/New)':<20} | {'Label (Old/New)':<15}")
    print("-" * 85)

    for idx, row in df.iterrows():
        try:
            ticker = row.get('_ticker', row.get('ticker'))
            analysis_date = pd.to_datetime(row.get('_date', row.get('analysis_date')))
            
            # Try to find DTE
            dte = row.get('dte', 30)
            if 'butterfly' in row and isinstance(row['butterfly'], dict):
                 dte = row['butterfly'].get('dte', 30)
            
            # Old Evaluation (Fixed 14 days)
            old_eval_date = analysis_date + timedelta(days=14)
            # Check if old ROI exists in debug column or re-calculate (skipped for speed in test mode usually, but useful for comparison)
            old_roi = row.get('_debug_roi', row.get('roi_14d', 0.0))
            old_label = row.get('label', -1)

            # New Dynamic Evaluation
            new_eval_date, days_held = calculate_dynamic_evaluation_date(analysis_date, dte)
            
            # Fetch Price
            future_data = yf.download(
                ticker, 
                start=new_eval_date, 
                end=new_eval_date + timedelta(days=5), 
                progress=False
            )

            if len(future_data) == 0:
                print(f"  [Warn] No data for {ticker} at {new_eval_date.date()}")
                failed += 1
                continue

            # Handle potentially multi-level columns or single series
            close_data = future_data['Close']
            if isinstance(close_data, pd.DataFrame):
                future_price = float(close_data.iloc[0, 0])
            else:
                future_price = float(close_data.iloc[0])
            
            # Try to get strikes from row, or fallback to butterfly dict
            if 'lower_strike' in row:
                lower = row['lower_strike']
                center = row['center_strike']
                upper = row['upper_strike']
                max_profit = row.get('max_profit', 1.0)
                cost = row.get('max_loss', row.get('net_debit', 1.0))
            else:
                # Try to extract from butterfly dict or full_result
                butterfly = row.get('butterfly', {})
                if not butterfly and 'full_result' in row:
                    import json
                    try:
                        fr = json.loads(row['full_result']) if isinstance(row['full_result'], str) else row['full_result']
                        butterfly = fr.get('butterfly', {})
                    except:
                        pass
                
                lower = butterfly.get('lower_strike')
                center = butterfly.get('center_strike')
                upper = butterfly.get('upper_strike')
                max_profit = butterfly.get('max_profit', 1.0)
                cost = butterfly.get('max_loss', butterfly.get('net_debit', 1.0))

            if lower is None or center is None or upper is None:
                print(f"  [Error] Missing strike data for {ticker}")
                failed += 1
                continue

            new_roi = calculate_butterfly_roi(future_price, lower, center, upper, cost, max_profit)
            new_label = classify_roi(new_roi)
            
            # Store result
            results.append({
                'original_idx': idx,
                'roi_dynamic': new_roi,
                'label': new_label,
                'days_held': days_held
            })
            
            # Print Comparison
            print(f"{ticker:<8} | {dte:<4} | {str(old_eval_date.date())}/{str(new_eval_date.date())} | {old_roi:>7.1%}/{new_roi:>7.1%} | {old_label:>5}/{new_label:>5}")
            
            successful += 1

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    logger.info(f"\nProcessing Complete. Success: {successful}, Failed: {failed}")

    # If applying changes
    if apply_changes and not limit:
        logger.info("Applying changes to dataset...")
        for res in results:
            idx = res['original_idx']
            df.at[idx, 'label'] = res['label']
            df.at[idx, 'roi_dynamic'] = res['roi_dynamic']
            # Update _debug_roi if needed or keep both
            # df.at[idx, '_debug_roi'] = res['roi_dynamic'] 
        
        output_file = output_path or input_path.replace('.parquet', '_relabeled.parquet')
        df.to_parquet(output_file, index=False)
        logger.info(f"✅ Saved relabeled data to: {output_file}")
    
    elif limit:
        logger.info("Test run completed. No files saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix ML Label Bias")
    parser.add_argument('--input', type=str, default='ml/training_data_deep.parquet', help='Path to input parquet file')
    parser.add_argument('--output', type=str, default=None, help='Path to output parquet file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples for testing')
    parser.add_argument('--apply', action='store_true', help='Apply changes and save file')
    
    args = parser.parse_args()
    
    # If not applying, default to limit 5 for safety unless specified
    if not args.apply and args.limit is None:
        args.limit = 5
        
    process_dataset(args.input, args.output, args.limit, args.apply)

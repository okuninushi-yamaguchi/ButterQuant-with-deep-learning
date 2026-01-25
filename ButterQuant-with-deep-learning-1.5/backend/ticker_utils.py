# -*- coding: utf-8 -*-
"""
Ticker Utilities - 股票代码工具库
提供从 Markdown 文件加载、过滤和标记代码的功能 / Provides ticker loading, filtering, and tagging from Markdown files
"""
import os

def load_tickers_from_markdown(file_path):
    """
    从Markdown文件加载股票代码 / Load tickers from Markdown file
    假设格式为每行一个代码 / Assume one ticker per line
    """
    tickers = []
    if not os.path.exists(file_path):
        print(f"Warning: Ticker file not found at {file_path} / File not found")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 忽略Markdown标题和列表标记 / Ignore Markdown headers and list markers
        if line.startswith('#'):
            continue
        if line.startswith('- '):
            line = line[2:]
        if line.startswith('* '):
            line = line[2:]
            
        # 提取代码 / Extract ticker
        # 处理 "AAPL - Apple Inc." 格式 / Handle "AAPL - Apple Inc." format
        parts = line.split(' ')
        if parts:
            ticker = parts[0].strip().upper()
            # 简单的验证：纯字母且长度在1-5之间 (美股)
            if ticker.isalpha() and 1 <= len(ticker) <= 5:
                tickers.append(ticker)
                
    # 去重 / De-duplicate
    return list(set(tickers))

def get_tickers_with_tags(file_map):
    """
    加载 Ticker 并附带分类标签 / Load tickers with category tags
    file_map: {'NAS100': 'path/to/nas100.md', ...}
    Returns: {'AAPL': ['NAS100'], ...}
    """
    ticker_tags = {}
    
    for tag, path in file_map.items():
        tickers = load_tickers_from_markdown(path)
        for t in tickers:
            if t not in ticker_tags:
                ticker_tags[t] = []
            if tag not in ticker_tags[t]:
                ticker_tags[t].append(tag)
                
    return ticker_tags

def merge_ticker_lists(files):
    """合并多个来源的Ticker / Merge tickers from multiple sources"""
    all_tickers = set()
    for f in files:
        tickers = load_tickers_from_markdown(f)
        all_tickers.update(tickers)
    return sorted(list(all_tickers))

if __name__ == "__main__":
    # 模块测试 / Module Test
    t = load_tickers_from_markdown('../nas100.md')
    print(f"Loaded {len(t)} tickers from nas100.md")

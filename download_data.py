#!/usr/bin/env python3
"""Download DIA and QQQ data in the same format as BRK-B."""
import json
import os
import pandas as pd
import yfinance as yf


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(df, path):
    """Save DataFrame to CSV with BOM for Excel compatibility, matching BRK-B format."""
    if df is None or df.empty:
        # Create empty file with just a header comment
        with open(path, 'w', encoding='utf-8-sig') as f:
            f.write('')
        return
    # BRK-B uses utf-8-sig (BOM) and index=True
    df.to_csv(path, encoding='utf-8-sig')


def download_ticker(ticker_symbol, folder_name):
    print(f"\n{'='*50}")
    print(f"Downloading {ticker_symbol} -> {folder_name}/")
    print(f"{'='*50}")
    ensure_dir(folder_name)
    prefix = folder_name.replace('_data', '')

    t = yf.Ticker(ticker_symbol)

    # ===== Price data =====
    print(f"[{ticker_symbol}] Downloading price histories...")
    try:
        daily = t.history(period="max")
        if not daily.empty:
            daily.index.name = 'Date'
            daily.columns = [c.replace('Stock Splits', 'Stock Splits').replace('Capital Gains', 'Capital Gains') for c in daily.columns]
            # Rename to match BRK-B column names exactly
            rename_map = {
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume',
                'Dividends': 'Dividends',
                'Stock Splits': 'Stock Splits'
            }
            daily = daily.rename(columns=rename_map)
            save_csv(daily, os.path.join(folder_name, f"{prefix}_daily.csv"))
            print(f"  -> {prefix}_daily.csv ({len(daily)} rows)")
    except Exception as e:
        print(f"  daily ERROR: {e}")

    try:
        hourly = t.history(period="2y", interval="1h")
        if not hourly.empty:
            hourly.index.name = 'Datetime'
            save_csv(hourly, os.path.join(folder_name, f"{prefix}_1h_2y.csv"))
            print(f"  -> {prefix}_1h_2y.csv ({len(hourly)} rows)")
    except Exception as e:
        print(f"  hourly ERROR: {e}")

    try:
        d2024 = t.history(start="2024-01-01")
        if not d2024.empty:
            d2024.index.name = 'Date'
            save_csv(d2024, os.path.join(folder_name, f"{prefix}_daily_2024_now.csv"))
            print(f"  -> {prefix}_daily_2024_now.csv ({len(d2024)} rows)")
    except Exception as e:
        print(f"  daily2024 ERROR: {e}")

    try:
        w2024 = t.history(start="2024-01-01", interval="1wk")
        if not w2024.empty:
            w2024.index.name = 'Date'
            save_csv(w2024, os.path.join(folder_name, f"{prefix}_weekly_2024_now.csv"))
            print(f"  -> {prefix}_weekly_2024_now.csv ({len(w2024)} rows)")
    except Exception as e:
        print(f"  weekly2024 ERROR: {e}")

    try:
        m2024 = t.history(start="2024-01-01", interval="1mo")
        if not m2024.empty:
            m2024.index.name = 'Date'
            save_csv(m2024, os.path.join(folder_name, f"{prefix}_monthly_2024_now.csv"))
            print(f"  -> {prefix}_monthly_2024_now.csv ({len(m2024)} rows)")
    except Exception as e:
        print(f"  monthly2024 ERROR: {e}")

    # ===== Info =====
    print(f"[{ticker_symbol}] Downloading info...")
    try:
        info = t.info
        if info:
            save_json(info, os.path.join(folder_name, f"{prefix}_info.json"))
            print(f"  -> {prefix}_info.json ({len(info)} keys)")
    except Exception as e:
        print(f"  info ERROR: {e}")

    # ===== Fundamentals (ETFs usually don't have these) =====
    fundamentals = {
        'balance_sheet': 'balance_sheet',
        'quarterly_balance_sheet': 'quarterly_balance_sheet',
        'income_stmt': 'income_statement',
        'quarterly_income_stmt': 'quarterly_income_statement',
        'cashflow': 'cashflow',
        'quarterly_cashflow': 'quarterly_cashflow',
        'institutional_holders': 'institutional_holders',
        'major_holders': 'major_holders',
        'earnings_estimate': 'earnings_estimate',
        'recommendations': 'recommendations',
    }

    for attr, fname in fundamentals.items():
        print(f"[{ticker_symbol}] Downloading {attr}...")
        try:
            df = getattr(t, attr)
            if df is not None and not df.empty:
                save_csv(df, os.path.join(folder_name, f"{prefix}_{fname}.csv"))
                print(f"  -> {prefix}_{fname}.csv ({df.shape})")
            else:
                # Save empty file to keep format consistent
                save_csv(df, os.path.join(folder_name, f"{prefix}_{fname}.csv"))
                print(f"  -> {prefix}_{fname}.csv (EMPTY - ETFs don't have this data)")
        except Exception as e:
            print(f"  {attr} ERROR: {e}")

    # ===== Splits =====
    print(f"[{ticker_symbol}] Downloading splits...")
    try:
        splits = t.splits
        if splits is not None and not splits.empty:
            splits_df = splits.to_frame(name='Stock Splits')
            splits_df.index.name = 'Date'
            save_csv(splits_df, os.path.join(folder_name, f"{prefix}_splits.csv"))
            print(f"  -> {prefix}_splits.csv ({len(splits_df)} rows)")
        else:
            save_csv(pd.DataFrame(), os.path.join(folder_name, f"{prefix}_splits.csv"))
            print(f"  -> {prefix}_splits.csv (EMPTY)")
    except Exception as e:
        print(f"  splits ERROR: {e}")

    print(f"Done with {ticker_symbol}!")


if __name__ == '__main__':
    os.chdir(r'D:\0521数据清洗')
    download_ticker('DIA', 'dj_data')
    download_ticker('QQQ', 'nasdaq_data')
    print("\n" + "="*50)
    print("All downloads complete!")
    print("="*50)

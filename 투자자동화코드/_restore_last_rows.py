"""최신 raw 데이터에서 features.csv에 누락된 날짜 행 복원"""
import pandas as pd, numpy as np, json, os
import ta as ta_lib

with open('models/feature_columns.json', encoding='utf-8') as f:
    FEAT_COLS = json.load(f)

raw = pd.read_csv('data/ohlcv_raw_kr.csv')
raw['date'] = pd.to_datetime(raw['date'])
# ticker를 6자리 문자열로 통일
raw['ticker'] = raw['ticker'].astype(str).str.zfill(6)

feat_existing = pd.read_csv('data/features.csv')
feat_existing['date'] = pd.to_datetime(feat_existing['date'])
feat_existing['ticker'] = feat_existing['ticker'].astype(str)

kr_mask = feat_existing['asset_type'] != 'us_stocks'
latest_feat_date = feat_existing[kr_mask]['date'].max()
latest_raw_date  = raw['date'].max()
print(f'features KR 최신: {latest_feat_date.date()}, raw 최신: {latest_raw_date.date()}')

if latest_raw_date <= latest_feat_date:
    print('[OK] 추가할 새 날짜 없음 - features 이미 최신')
    exit()

new_rows = []
for ticker in raw['ticker'].unique():
    tdf = raw[raw['ticker'] == ticker].copy().sort_values('date').reset_index(drop=True)
    if len(tdf) < 60:
        continue

    existing_ticker = feat_existing[feat_existing['ticker'] == ticker]
    if existing_ticker.empty:
        continue
    atype = existing_ticker['asset_type'].iloc[-1]

    # 기술적 지표 계산
    tdf['ma5']  = ta_lib.trend.sma_indicator(tdf['close'], window=5)
    tdf['ma20'] = ta_lib.trend.sma_indicator(tdf['close'], window=20)
    tdf['ma60'] = ta_lib.trend.sma_indicator(tdf['close'], window=60)
    tdf['rsi14'] = ta_lib.momentum.RSIIndicator(tdf['close'], window=14).rsi()
    macd_obj = ta_lib.trend.MACD(tdf['close'], window_slow=26, window_fast=12, window_sign=9)
    tdf['macd'] = macd_obj.macd()
    tdf['macd_signal'] = macd_obj.macd_signal()
    tdf['macd_hist'] = macd_obj.macd_diff()
    bb_obj = ta_lib.volatility.BollingerBands(tdf['close'], window=20, window_dev=2)
    tdf['bb_upper'] = bb_obj.bollinger_hband()
    tdf['bb_mid']   = bb_obj.bollinger_mavg()
    tdf['bb_lower'] = bb_obj.bollinger_lband()
    tdf['bb_width'] = (tdf['bb_upper'] - tdf['bb_lower']) / tdf['bb_mid']
    tdf['atr14'] = ta_lib.volatility.AverageTrueRange(
        tdf['high'], tdf['low'], tdf['close'], window=14).average_true_range()
    tdf['vol_ma20'] = ta_lib.trend.sma_indicator(tdf['volume'], window=20)
    tdf['vol_ratio'] = tdf['volume'] / tdf['vol_ma20']
    tdf['close_ma5_ratio']  = tdf['close'] / tdf['ma5']  - 1
    tdf['close_ma20_ratio'] = tdf['close'] / tdf['ma20'] - 1
    tdf['close_ma60_ratio'] = tdf['close'] / tdf['ma60'] - 1
    tdf['ma5_ma20_ratio']   = tdf['ma5']   / tdf['ma20'] - 1
    tdf['ret_1d'] = tdf['close'].pct_change(1)
    tdf['ret_3d'] = tdf['close'].pct_change(3)
    tdf['ret_5d'] = tdf['close'].pct_change(5)
    tdf['asset_type'] = atype
    tdf['next_ret'] = tdf['close'].pct_change(1).shift(-1)
    tdf['label'] = tdf['next_ret'].apply(
        lambda r: 2 if r >= 0.01 else (0 if r <= -0.01 else 1) if pd.notna(r) else np.nan)

    new = tdf[tdf['date'] > latest_feat_date].copy()
    if new.empty:
        continue

    for _, row in new.iterrows():
        row_d = row.to_dict()
        for c in FEAT_COLS:
            if c in row_d and pd.isna(row_d[c]):
                row_d[c] = 0.0
        new_rows.append(row_d)

if new_rows:
    new_df = pd.DataFrame(new_rows)
    for c in feat_existing.columns:
        if c not in new_df.columns:
            new_df[c] = np.nan
    new_df = new_df[[c for c in feat_existing.columns if c in new_df.columns]]

    combined = pd.concat([feat_existing, new_df], ignore_index=True)
    combined.to_csv('data/features.csv', index=False)
    added_dates = sorted(new_df['date'].unique())
    print(f'[OK] {len(new_rows)}행 추가 -> features.csv ({len(combined)}행)')
    print(f'     추가된 날짜: {added_dates}')
else:
    print('[OK] 추가된 행 없음')

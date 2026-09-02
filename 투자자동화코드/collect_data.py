"""
collect_data.py - 다자산 확장 수집 스크립트

수집 대상:
    - 코스피 대형주 / 국내 ETF / KRX 금현물  → KIS API  → features_kr.csv
    - 미국 주식                               → yfinance → features_us.csv
    - 전체 병합                               → features.csv

실행:
    python collect_data.py
"""

import os, time, json, requests
import pandas as pd
import numpy as np
import ta as ta_lib                       # pandas_ta 대신 ta 라이브러리 사용
from datetime import datetime, timedelta
from config import (
    APP_KEY, APP_SECRET, BASE_URL,
    TICKERS, TICKERS_US, ASSETS, FETCH_MACRO,
    DATA_DIR, MODEL_DIR,
)

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False
    print("[경고] yfinance 미설치 - 미국 주식 수집 생략 (pip install yfinance)")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TOKEN_CACHE_FILE = os.path.join(DATA_DIR, ".token_cache.json")


# ─────────────────────────────────────────────
# 공통: 기술적 지표 + 레이블 생성
# ─────────────────────────────────────────────
def add_features(df: pd.DataFrame, asset_type: str = "stock") -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)

    # 이동평균
    df["ma5"]  = ta_lib.trend.sma_indicator(df["close"], window=5)
    df["ma20"] = ta_lib.trend.sma_indicator(df["close"], window=20)
    df["ma60"] = ta_lib.trend.sma_indicator(df["close"], window=60)

    # RSI
    df["rsi14"] = ta_lib.momentum.RSIIndicator(df["close"], window=14).rsi()

    # MACD
    macd_obj = ta_lib.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"]        = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"]   = macd_obj.macd_diff()

    # 볼린저밴드
    bb_obj = ta_lib.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb_obj.bollinger_hband()
    df["bb_mid"]   = bb_obj.bollinger_mavg()
    df["bb_lower"] = bb_obj.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # ATR (변동성)
    df["atr14"]     = ta_lib.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

    # 거래량 지표
    df["vol_ma20"]  = ta_lib.trend.sma_indicator(df["volume"], window=20)
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]

    # 파생 피처
    df["close_ma5_ratio"]  = df["close"] / df["ma5"]  - 1
    df["close_ma20_ratio"] = df["close"] / df["ma20"] - 1
    df["close_ma60_ratio"] = df["close"] / df["ma60"] - 1
    df["ma5_ma20_ratio"]   = df["ma5"]   / df["ma20"] - 1

    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_3d"] = df["close"].pct_change(3)
    df["ret_5d"] = df["close"].pct_change(5)

    # 자산군 레이블
    df["asset_type"] = asset_type

    # 정답 레이블 (+1% 매수=2, -1% 매도=0, 나머지 관망=1)
    df["next_ret"] = df["close"].pct_change(1).shift(-1)
    df["label"]    = df["next_ret"].apply(
        lambda r: 2 if r >= 0.01 else (0 if r <= -0.01 else 1)
    )
    return df


# ─────────────────────────────────────────────
# 1. KIS API - 국내 (코스피 + ETF + 금현물)
# ─────────────────────────────────────────────
def get_access_token() -> str:
    """토큰 캐시 활용 - 당일 발급된 토큰이 있으면 재사용 (KIS 1일 1회 발급 정책)"""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(TOKEN_CACHE_FILE):
        with open(TOKEN_CACHE_FILE, encoding="utf-8") as f:
            cache = json.load(f)
        issued_at = datetime.fromisoformat(cache.get("issued_at", "2000-01-01"))
        if (datetime.now() - issued_at).total_seconds() < 82800:  # 23시간 이내
            print(f"[인증] 캐시된 토큰 재사용 (발급: {issued_at.strftime('%H:%M')})")
            return cache["access_token"]

    resp = requests.post(
        f"{BASE_URL}/oauth2/tokenP",
        json={"grant_type": "client_credentials", "appkey": APP_KEY, "appsecret": APP_SECRET},
        timeout=10,
    )
    resp.raise_for_status()
    token = resp.json().get("access_token")
    if not token:
        raise ValueError(f"토큰 발급 실패: {resp.json()}")

    with open(TOKEN_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"access_token": token, "issued_at": datetime.now().isoformat()}, f)
    print("[인증] KIS 액세스 토큰 신규 발급 완료")
    return token


def fetch_ohlcv_kis(token: str, ticker: str, days: int = 400) -> pd.DataFrame:
    """KIS API 일봉 OHLCV 수집 (100일씩 분할, 500 오류 스킵)"""
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHKST03010100",
    }
    all_rows = []
    end_date = datetime.today()

    for _ in range((days // 100) + 1):
        start_date = end_date - timedelta(days=100)
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": start_date.strftime("%Y%m%d"),
            "FID_INPUT_DATE_2": end_date.strftime("%Y%m%d"),
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "0",
        }
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 500:
            print(f"  [경고] {ticker} 500오류, 해당 구간 건너뜀")
            end_date = start_date - timedelta(days=1)
            time.sleep(2.0)
            continue
        if resp.status_code != 200:
            print(f"  [경고] {ticker} 조회 실패: {resp.status_code}")
            break
        data = resp.json().get("output2", [])
        if not data:
            break
        for row in data:
            all_rows.append({
                "date":   row.get("stck_bsop_date"),
                "ticker": ticker,
                "open":   float(row.get("stck_oprc", 0)),
                "high":   float(row.get("stck_hgpr", 0)),
                "low":    float(row.get("stck_lwpr", 0)),
                "close":  float(row.get("stck_clpr", 0)),
                "volume": float(row.get("acml_vol", 0)),
            })
        end_date = start_date - timedelta(days=1)
        time.sleep(1.0)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    # 거래량 0 행 제거 (시장 미개장일 / API 미확정 데이터)
    df = df[df["volume"] > 0].reset_index(drop=True)
    return df


def load_existing_raw(path: str) -> pd.DataFrame:
    """기존 원본 CSV 로드. 없으면 빈 DataFrame 반환."""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"ticker": str})
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df["date"]   = pd.to_datetime(df["date"])
    return df


def collect_domestic(token: str) -> pd.DataFrame:
    """코스피 대형주 + ETF + 금현물 증분 수집"""
    raw_path = os.path.join(DATA_DIR, "ohlcv_raw_kr.csv")
    existing_raw = load_existing_raw(raw_path)

    # 자산군 구분자 매핑
    ticker_to_type = {}
    for atype, tlist in ASSETS.items():
        if atype == "us_stocks":
            continue
        for t in tlist:
            ticker_to_type[t] = atype

    all_raw_parts  = []
    failed_tickers = []

    for ticker in TICKERS:
        ticker_zfill = str(ticker).zfill(6)
        atype = ticker_to_type.get(ticker, "kospi_large")

        if not existing_raw.empty and ticker_zfill in existing_raw["ticker"].values:
            ticker_rows = existing_raw[existing_raw["ticker"] == ticker_zfill]
            latest = ticker_rows["date"].max()
            if len(ticker_rows) < 200:
                days_needed = 400
                print(f"  [{atype}] {ticker} (이력 부족 {len(ticker_rows)}행, 전체 재수집) ...", end=" ")
            else:
                days_needed = 70
                print(f"  [{atype}] {ticker} (기존 최신: {latest.strftime('%Y-%m-%d')}) ...", end=" ")
        else:
            days_needed = 400
            print(f"  [{atype}] {ticker} (신규) ...", end=" ")

        raw = fetch_ohlcv_kis(token, ticker_zfill, days=days_needed)
        if raw.empty:
            print("없음, 기존 데이터 유지")
            failed_tickers.append(ticker_zfill)
            time.sleep(2.0)
            continue
        print(f"{len(raw)}행")

        # 기존 원본 데이터와 병합 - 신규 행만 추가
        if not existing_raw.empty and ticker_zfill in existing_raw["ticker"].values:
            existing_ticker = existing_raw[existing_raw["ticker"] == ticker_zfill].copy()
            combined_raw = (
                pd.concat([existing_ticker, raw])
                .drop_duplicates("date")
                .sort_values("date")
                .reset_index(drop=True)
            )
        else:
            combined_raw = raw

        all_raw_parts.append(combined_raw)
        time.sleep(1.5)

    # 원본 CSV 저장 (실패 종목은 기존 유지)
    if all_raw_parts:
        new_raw = pd.concat(all_raw_parts, ignore_index=True)
        new_raw["ticker"] = new_raw["ticker"].astype(str).str.zfill(6)
        updated = new_raw["ticker"].unique()

        if not existing_raw.empty:
            kept = existing_raw[~existing_raw["ticker"].isin(updated)]
            final_raw = pd.concat([kept, new_raw], ignore_index=True).sort_values(
                ["ticker", "date"]
            ).reset_index(drop=True)
        else:
            final_raw = new_raw

        final_raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
        print(f"  -> {raw_path} ({len(final_raw)}행)")
    else:
        final_raw = existing_raw

    if failed_tickers:
        print(f"  [주의] API 실패: {failed_tickers}")

    # 피처 계산 (종목별 전체 기간 기준)
    all_feat = []
    for ticker_zfill in final_raw["ticker"].unique():
        tdf = final_raw[final_raw["ticker"] == ticker_zfill].copy()
        atype = ticker_to_type.get(ticker_zfill.lstrip("0") or ticker_zfill, "kospi_large")
        # 6자리로 zfill한 티커로도 찾기
        for orig_ticker, at in ticker_to_type.items():
            if str(orig_ticker).zfill(6) == ticker_zfill:
                atype = at
                break
        feat = add_features(tdf, asset_type=atype)
        # dropna(subset=FEAT_COLS): 초기 NaN 구간 제거, 마지막 행(next_ret NaN) 보존
        _fc_path = os.path.join(MODEL_DIR, "feature_columns.json")
        if os.path.exists(_fc_path):
            with open(_fc_path, encoding="utf-8") as _f:
                _feat_cols = json.load(_f)
            feat_clean = feat.dropna(subset=_feat_cols)
            # 마지막 행(최신 날짜)이 feat_clean에 없을 때만 추가 (next_ret NaN으로 제외된 경우)
            last_date = feat["date"].max()
            if last_date not in feat_clean["date"].values:
                last_row = feat[feat["date"] == last_date].copy()
                for _c in _feat_cols:
                    if _c in last_row.columns and last_row[_c].isna().any():
                        last_row[_c] = 0.0
                feat_clean = pd.concat([feat_clean, last_row], ignore_index=True)
            all_feat.append(feat_clean)
        else:
            all_feat.append(feat.dropna())

    return pd.concat(all_feat, ignore_index=True) if all_feat else pd.DataFrame()


# ─────────────────────────────────────────────
# 2. yfinance - 미국 주식 + 매크로 피처
# ─────────────────────────────────────────────
def _yf_download_to_df(ticker: str, period: str = "2y") -> pd.DataFrame:
    """yfinance 다운로드 + MultiIndex 컬럼 정규화"""
    raw = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()
    # MultiIndex (yfinance 0.2+) 처리
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0].lower() for col in raw.columns]
    else:
        raw.columns = [col.lower() for col in raw.columns]
    df = raw.reset_index()
    # Date 컬럼 이름 정규화
    df = df.rename(columns={"Date": "date", "index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def fetch_macro_features() -> pd.DataFrame:
    """USD/KRW, VIX, 나스닥, S&P500, 금 선물 수집"""
    print("\n[매크로] 환율·VIX·나스닥 수집 중...")
    macro_tickers = {
        "USDKRW=X": "usdkrw",
        "^VIX":     "vix",
        "^IXIC":    "nasdaq",
        "^GSPC":    "sp500",
        "GC=F":     "gold_usd",
    }
    macro_df = None
    for yticker, col in macro_tickers.items():
        try:
            df = _yf_download_to_df(yticker)
            if df.empty:
                continue
            s = df.set_index("date")["close"].rename(col)
            macro_df = s.to_frame() if macro_df is None else pd.concat([macro_df, s], axis=1)
        except Exception as e:
            print(f"  {yticker} 수집 실패: {e}")

    if macro_df is None:
        return pd.DataFrame()

    macro_df = macro_df.reset_index()
    macro_df["date"] = pd.to_datetime(macro_df["date"]).dt.normalize()

    if "usdkrw" in macro_df.columns:
        macro_df["usdkrw_ret1d"] = macro_df["usdkrw"].pct_change(1)
    if "nasdaq" in macro_df.columns:
        macro_df["nasdaq_ret1d"] = macro_df["nasdaq"].pct_change(1)
    if "vix" in macro_df.columns:
        macro_df["vix_ma5"] = macro_df["vix"].rolling(5).mean()

    print(f"  매크로 피처 수집 완료: {len(macro_df)}행")
    return macro_df


def collect_us_stocks(macro_df: pd.DataFrame) -> pd.DataFrame:
    """미국 주식 수집 + 매크로 피처 병합"""
    if not YFINANCE_OK:
        return pd.DataFrame()

    all_feat = []
    for ticker in TICKERS_US:
        print(f"  [us_stocks] {ticker} ...", end=" ")
        try:
            df = _yf_download_to_df(ticker)
            if df.empty:
                print("없음")
                continue
            df["ticker"] = ticker
            # 필요 컬럼 확인
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    df[col] = np.nan

            # 매크로 피처 병합
            if not macro_df.empty:
                df = df.merge(macro_df, on="date", how="left")

            feat = add_features(df, asset_type="us_stocks")
            # dropna(subset=FEAT_COLS): 초기 NaN 제거, 마지막 행(next_ret NaN) 보존
            _fc_path = os.path.join(MODEL_DIR, "feature_columns.json")
            if os.path.exists(_fc_path):
                with open(_fc_path, encoding="utf-8") as _f:
                    _feat_cols = json.load(_f)
                feat_clean = feat.dropna(subset=_feat_cols)
                last_date = feat["date"].max()
                if last_date not in feat_clean["date"].values:
                    last_row = feat[feat["date"] == last_date].copy()
                    for _c in _feat_cols:
                        if _c in last_row.columns and last_row[_c].isna().any():
                            last_row[_c] = 0.0
                    feat_clean = pd.concat([feat_clean, last_row], ignore_index=True)
                all_feat.append(feat_clean)
            else:
                all_feat.append(feat.dropna())
            print(f"{len(df)}행")
        except Exception as e:
            print(f"오류: {e}")
        time.sleep(0.3)

    return pd.concat(all_feat, ignore_index=True) if all_feat else pd.DataFrame()


# ─────────────────────────────────────────────
# 3. 상위 n% 필터 유틸
# ─────────────────────────────────────────────
def filter_top_n(result_df: pd.DataFrame,
                 top_n_percent: float = 0.3,
                 min_confidence: float = 0.60) -> pd.DataFrame:
    df = result_df.copy()
    df["conf_float"] = df["신뢰도"].str.rstrip("%").astype(float) / 100
    buy_df = df[(df["신호"] == "매수") & (df["conf_float"] >= min_confidence)]
    if buy_df.empty:
        return buy_df
    n_keep = max(1, int(len(buy_df) * top_n_percent))
    return buy_df.nlargest(n_keep, "conf_float").reset_index(drop=True)


# ─────────────────────────────────────────────
# 4. 전체 실행
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("다자산 데이터 수집 시작")
    print("=" * 55)

    # 국내 수집 (KIS API)
    print(f"\n[1/3] 국내 종목 수집 (KIS API) - {len(TICKERS)}종목")
    token  = get_access_token()
    kr_df  = collect_domestic(token)
    if not kr_df.empty:
        path = os.path.join(DATA_DIR, "features_kr.csv")
        kr_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"-> {path} ({len(kr_df)}행)")

    # 매크로 피처 (yfinance)
    macro_df = pd.DataFrame()
    if FETCH_MACRO and YFINANCE_OK:
        print("\n[2/3] 매크로 피처 수집 (yfinance)")
        macro_df = fetch_macro_features()
    else:
        print("\n[2/3] 매크로 피처 수집 생략")

    # 미국 주식 (yfinance)
    print(f"\n[3/3] 미국 주식 수집 (yfinance) - {len(TICKERS_US)}종목")
    us_df = collect_us_stocks(macro_df)
    if not us_df.empty:
        path = os.path.join(DATA_DIR, "features_us.csv")
        us_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"-> {path} ({len(us_df)}행)")

    # 전체 병합 → features.csv
    frames = [f for f in [kr_df, us_df] if not f.empty]
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        path = os.path.join(DATA_DIR, "features.csv")
        all_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n[완료] 전체 병합 -> {path} ({len(all_df)}행)")
        if "asset_type" in all_df.columns:
            print(f"자산군별 행 수:\n{all_df['asset_type'].value_counts()}")
        print(f"레이블 분포:\n{all_df['label'].value_counts().sort_index()}")
    else:
        print("[경고] 수집된 데이터 없음")


if __name__ == "__main__":
    main()

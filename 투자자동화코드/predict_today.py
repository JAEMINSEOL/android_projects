"""
predict_today.py — 저장된 모델로 오늘 매매 신호 예측

실행 방법:
    python predict_today.py

사전 조건:
    1. collect_data.py 실행 완료 (data/features.csv 존재)
    2. train_model.py 실행 완료 (models/ 폴더 존재)

출력:
    - 터미널: 전체 종목 신호 + 최종 포트폴리오
    - data/today_signal_YYYYMMDD.csv  — 전체 신호
    - data/portfolio_YYYYMMDD.csv     — 상위 n% 필터 결과 (리포트 붙여넣기용)
"""

import os, json, joblib
import pandas as pd
import numpy as np
from datetime import datetime, date
import xgboost as xgb

from config import DATA_DIR, MODEL_DIR, TICKERS, TOP_N_PERCENT, MIN_CONFIDENCE
from collect_data_extended import filter_top_n

LABEL_MAP = {0: "매도", 1: "관망", 2: "매수"}


# ─────────────────────────────────────────────
# 1. 모델 로드 (변경 없음)
# ─────────────────────────────────────────────
def load_model():
    model_path  = os.path.join(MODEL_DIR, "xgb_model.json")
    feat_path   = os.path.join(MODEL_DIR, "feature_columns.json")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    meta_path   = os.path.join(MODEL_DIR, "model_meta.json")

    for p in [model_path, feat_path, scaler_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} 없음 — train_model.py 를 먼저 실행하세요.")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    with open(feat_path, encoding="utf-8") as f:
        feature_cols = json.load(f)

    scaler = joblib.load(scaler_path)

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    print(f"[모델 로드] 훈련일: {meta['trained_at']} | CV 정확도: {meta['cv_accuracy_mean']:.4f}")
    return model, feature_cols, scaler


# ─────────────────────────────────────────────
# 2. 예측 실행
# ─────────────────────────────────────────────
def predict_all(model, feature_cols, scaler) -> pd.DataFrame:

    feat_path = os.path.join(DATA_DIR, "features.csv")
    df = pd.read_csv(feat_path, parse_dates=["date"])
    df = df.sort_values(["ticker", "date"])

    results = []

    for ticker in TICKERS:
        tdf = df[df["ticker"] == ticker].copy()
        if tdf.empty or len(tdf) < 2:
            continue

        last  = tdf.iloc[-1]
        X_raw = last[feature_cols].values.reshape(1, -1)
        X_raw = np.where(np.isinf(X_raw), np.nan, X_raw)
        X_raw = np.nan_to_num(X_raw, nan=np.nanmedian(tdf[feature_cols].values))

        X_scaled = scaler.transform(X_raw)
        proba    = model.predict_proba(X_scaled)[0]
        pred_cls = int(np.argmax(proba))

        results.append({
            "ticker":        ticker,
            "자산군":        last.get("asset_type", "kospi_large"),
            "기준일":        last["date"].strftime("%Y-%m-%d"),
            "종가":          int(last["close"]),
            "신호":          LABEL_MAP[pred_cls],
            "신뢰도":        f"{proba[pred_cls]*100:.1f}%",
            "P_매도":        f"{proba[0]*100:.1f}%",
            "P_관망":        f"{proba[1]*100:.1f}%",
            "P_매수":        f"{proba[2]*100:.1f}%",
            "RSI":           f"{last.get('rsi14', 0):.1f}",
            "MACD히스토그램": f"{last.get('macd_hist', 0):.2f}",
            "BB위치":        f"{((last['close']-last.get('bb_lower',0))/(last.get('bb_upper',1)-last.get('bb_lower',0))*100):.0f}%",
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# 3. 결과 출력 및 저장
# ★ 변경점 4: top_df 파라미터 추가, 최종 포트폴리오 출력 블록 추가
# ─────────────────────────────────────────────
def print_and_save(result_df: pd.DataFrame, top_df: pd.DataFrame = None):
    today_str = date.today().strftime("%Y%m%d")

    if result_df.empty:
        print("예측 결과가 없습니다. 데이터를 확인하세요.")
        return

    # ── 전체 신호 출력 ──────────────────────────
    print("\n" + "=" * 70)
    print(f"  전체 종목 신호  ({date.today().strftime('%Y-%m-%d')})")
    print("=" * 70)
    print(f"{'티커':>8}  {'자산군':>12}  {'신호':>6}  {'신뢰도':>7}  {'종가':>9}  {'RSI':>5}")
    print("-" * 70)

    for _, row in result_df.iterrows():
        emoji = {"매수": "🟢", "매도": "🔴", "관망": "⚪"}.get(row["신호"], "")
        print(
            f"  {row['ticker']:>8}  {row['자산군']:>12}  "
            f"{emoji}{row['신호']:>4}  {row['신뢰도']:>7}  "
            f"{row['종가']:>9,}  {row['RSI']:>5}"
        )
    print("=" * 70)

    # ── 최종 포트폴리오 (상위 n% 필터 결과) ────
    if top_df is not None and not top_df.empty:
        print(f"\n{'━'*70}")
        print(f"  최종 포트폴리오 — 신뢰도 상위 {int(TOP_N_PERCENT*100)}%  "
              f"(하한 {int(MIN_CONFIDENCE*100)}% 이상, {len(top_df)}종목)")
        print(f"{'━'*70}")
        for i, row in top_df.iterrows():
            print(f"  {i+1}위  {row['ticker']:>8}  [{row['자산군']}]  "
                  f"신뢰도 {row['신뢰도']}  종가 {row['종가']:,}원")
        print(f"{'━'*70}")

        portfolio_path = os.path.join(DATA_DIR, f"portfolio_{today_str}.csv")
        top_df.to_csv(portfolio_path, index=False, encoding="utf-8-sig")
        print(f"\n[저장] 최종 포트폴리오 → {portfolio_path}")
    else:
        print("\n조건에 맞는 매수 후보가 없습니다. (관망 권장)")

    # ── 전체 신호 CSV 저장 ──────────────────────
    all_path = os.path.join(DATA_DIR, f"today_signal_{today_str}.csv")
    result_df.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"[저장] 전체 신호      → {all_path}")
    print("\n→ portfolio_*.csv 를 claude.ai 아침 리포트 프롬프트에 붙여넣으세요.\n")


# ─────────────────────────────────────────────
# 4. 실행
# ★ 변경점 3: filter_top_n 호출 추가
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("오늘 매매 신호 예측")
    print("=" * 50)

    model, feature_cols, scaler = load_model()
    result_df = predict_all(model, feature_cols, scaler)

    # ★ 여기가 추가된 두 줄
    top_df = filter_top_n(result_df, TOP_N_PERCENT, MIN_CONFIDENCE)

    print_and_save(result_df, top_df)


if __name__ == "__main__":
    main()

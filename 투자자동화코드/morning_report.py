"""
morning_report.py — 아침 자동 리포트 (스케줄러 실행용)

실행:
    python morning_report.py

동작:
    1. predict_today.py 예측 신호 생성
    2. data/my_portfolio.csv 현재 보유 현황 로드
    3. 보유 종목 손익 + 매매 신호 결합
    4. 수정 포트폴리오 리포트 생성 → data/portfolio_report_YYYYMMDD.md 저장
"""

import os, json, joblib
import pandas as pd
import numpy as np
from datetime import date, datetime
import xgboost as xgb

from config import DATA_DIR, MODEL_DIR, TICKERS, TICKERS_US

PORTFOLIO_FILE  = os.path.join(DATA_DIR, "my_portfolio.csv")
TOTAL_CASH      = 7_000_000   # 총 투자 가능 금액 (원) — 5M 기존 + 2M 추가
MIN_CASH_RATIO  = 0.20        # 최소 현금 비중
MAX_INVEST      = TOTAL_CASH * (1 - MIN_CASH_RATIO)   # 최대 투자 가능
MAX_SINGLE      = TOTAL_CASH * 0.25                    # 단일 종목 최대

TICKER_NAMES = {
    # 코스피 대형주
    "005930": "삼성전자",   "000660": "SK하이닉스", "005380": "현대차",
    "035420": "NAVER",     "051910": "LG화학",      "006400": "삼성SDI",
    "035720": "카카오",    "000270": "기아",         "105560": "KB금융",
    "055550": "신한지주",  "012330": "현대모비스",   "028260": "삼성물산",
    "066570": "LG전자",    "003550": "LG",           "017670": "SK텔레콤",
    "030200": "KT",        "096770": "SK이노베이션", "011200": "HMM",
    "010950": "S-Oil",     "086790": "하나금융지주",
    # 국내 ETF
    "069500": "KODEX 200",          "229200": "KODEX 코스닥150",
    "305720": "KODEX 2차전지산업",   "091160": "KODEX 반도체",
    "139220": "KODEX IT",            "114800": "KODEX 인버스",
    "252670": "KODEX 200선물인버스2X","148020": "KODEX 헬스케어",
    "364980": "KODEX 수소경제",      "379800": "KODEX 미국S&P500",
    # KRX 금현물
    "4GLD": "KRX 금현물",
    # 미국 주식
    "AAPL": "애플",   "MSFT": "MS",     "NVDA": "엔비디아", "GOOGL": "알파벳",
    "AMZN": "아마존", "META": "메타",   "TSLA": "테슬라",   "TSM":  "TSMC",
    "ASML": "ASML",   "AMD":  "AMD",
}

# 인버스 ETF: 시장 하락 헤지 전용 — 일반 매수/매도 신호와 섞이면 반대 의미
# 별도 섹션으로 표시하고, 학습에서도 제외됨 (train_model.py EXCLUDE_TICKERS)
INVERSE_ETF_TICKERS = {"114800", "252670"}


# ──────────────────────────────────────────────
# 1. 모델 로드 + 예측
# ──────────────────────────────────────────────
def run_prediction() -> pd.DataFrame:
    model_path  = os.path.join(MODEL_DIR, "xgb_model.json")
    feat_path   = os.path.join(MODEL_DIR, "feature_columns.json")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    with open(feat_path, encoding="utf-8") as f:
        feature_cols = json.load(f)
    scaler = joblib.load(scaler_path)

    df = pd.read_csv(
        os.path.join(DATA_DIR, "features.csv"),
        parse_dates=["date"],
        dtype={"ticker": str},
    )
    # KR 티커는 6자리 zfill, US 티커(알파벳)는 그대로 유지
    def normalize_ticker(t: str) -> str:
        return t.zfill(6) if t.isdigit() else t
    df["ticker"] = df["ticker"].apply(normalize_ticker)
    df = df.sort_values(["ticker", "date"])

    # KR 티커: 6자리 zfill / US 티커: 원본 그대로
    all_tickers = [t.zfill(6) for t in TICKERS] + list(TICKERS_US)

    rows = []
    for ticker in all_tickers:
        tdf = df[df["ticker"] == ticker]
        if len(tdf) < 2:
            continue
        last   = tdf.iloc[-1]
        # 모델이 학습한 피처만 사용 (없는 피처는 0으로 채움)
        available = [c for c in feature_cols if c in last.index]
        X_raw  = np.zeros((1, len(feature_cols)))
        for i, col in enumerate(feature_cols):
            if col in last.index:
                X_raw[0, i] = float(last[col]) if pd.notna(last[col]) else 0.0
        X_raw  = np.where(np.isinf(X_raw), np.nan, X_raw)
        X_raw  = np.nan_to_num(X_raw, nan=0.0)
        X_df   = pd.DataFrame(X_raw, columns=feature_cols)
        proba  = model.predict_proba(scaler.transform(X_df))[0]
        cls    = int(np.argmax(proba))
        label  = ["매도", "관망", "매수"][cls]
        bb_pos = (
            (last["close"] - last.get("bb_lower", 0))
            / max(last.get("bb_upper", 1) - last.get("bb_lower", 0), 1)
            * 100
        )
        rows.append({
            "ticker":      ticker,
            "company":     TICKER_NAMES.get(ticker, ticker),
            "기준일":      last["date"].strftime("%Y-%m-%d"),
            "종가":        int(last["close"]),
            "신호":        label,
            "신뢰도":      float(f"{proba[cls]*100:.1f}"),
            "P_매도":      float(f"{proba[0]*100:.1f}"),
            "P_관망":      float(f"{proba[1]*100:.1f}"),
            "P_매수":      float(f"{proba[2]*100:.1f}"),
            "RSI":         round(float(last.get("rsi14", 0)), 1),
            "BB위치":      f"{bb_pos:.0f}%",
            "bb_upper":    float(last.get("bb_upper", 0)),
            "bb_lower":    float(last.get("bb_lower", 0)),
            "atr14":       float(last.get("atr14", 0)),
        })

    sig_df = pd.DataFrame(rows)
    today_str = date.today().strftime("%Y%m%d")
    sig_df.to_csv(
        os.path.join(DATA_DIR, f"today_signal_{today_str}.csv"),
        index=False, encoding="utf-8-sig",
    )
    return sig_df


# ──────────────────────────────────────────────
# 예약매도가 계산 유틸
# ──────────────────────────────────────────────
def _tick(price: float) -> int:
    """한국 주식 호가 단위 적용 후 반환 (원 단위)"""
    if price >= 500_000:  tick = 1_000
    elif price >= 200_000: tick = 500
    elif price >= 50_000:  tick = 100
    elif price >= 20_000:  tick = 50
    elif price >= 5_000:   tick = 10
    elif price >= 2_000:   tick = 5
    else:                  tick = 1
    return int(round(price / tick) * tick)


def calc_sell_targets(avg_price: float, last_price: float,
                      rsi: float, bb_upper: float, atr14: float) -> dict:
    """
    보유 종목별 안전 예약매도가 3단계 산출

    Returns
    -------
    dict: 손절가, 1차_익절가, 안전_예약매도가, 각 비고
    """
    gain_pct = (last_price - avg_price) / avg_price if avg_price > 0 else 0.0

    # ── 손절가 ────────────────────────────────
    # 수익 5% 이상 → 매입가 보전 트레일링 스탑 (손실 0 보장)
    # 수익 2~5%    → 매입가 -1% (절반 보전)
    # 그 외        → 매입가 -3% 기본 손절
    if gain_pct >= 0.05:
        stop = avg_price          # 원금 보전
        stop_note = f"원금 보전 (현재 +{gain_pct*100:.1f}%)"
    elif gain_pct >= 0.02:
        stop = avg_price * 0.99   # -1% 완화 손절
        stop_note = f"완화 손절 -1% (현재 +{gain_pct*100:.1f}%)"
    else:
        stop = avg_price * 0.97   # 기본 -3% 손절
        stop_note = "기본 손절 -3%"

    # ── 1차 익절가 ────────────────────────────
    # RSI 구간별 차등 적용
    if rsi >= 75:
        mult1 = 1.005   # 과열 → 바로 익절
        tp1_note = "RSI 과열, 소폭 익절 +0.5%"
    elif rsi >= 60:
        mult1 = 1.015   # 고RSI → +1.5%
        tp1_note = "RSI 높음 +1.5%"
    elif rsi >= 40:
        mult1 = 1.02    # 중립 → +2%
        tp1_note = "표준 익절 +2%"
    else:
        mult1 = 1.03    # 저RSI → 반등 기대 +3%
        tp1_note = "저RSI 반등 기대 +3%"
    tp1 = last_price * mult1

    # ── 안전 예약매도가 ───────────────────────
    # BB 상단이 유효하면 기술적 저항선으로 사용
    # 단, BB 상단이 1차 익절보다 낮으면 BB 상단 우선 (이미 과열)
    if bb_upper > last_price * 1.001:         # BB 상단이 현재가보다 위
        if bb_upper < tp1:
            safe = bb_upper                    # BB 상단이 더 가까운 저항
            safe_note = f"BB 상단 저항선 ({int(bb_upper):,}원)"
        else:
            safe = tp1                         # 1차 익절이 더 보수적
            safe_note = tp1_note
    else:                                      # BB 상단 이미 돌파 (과열)
        safe = last_price * 1.01              # +1% 즉시 익절
        safe_note = "BB 상단 돌파 — +1% 즉시 익절"

    return {
        "손절가":           _tick(stop),
        "손절_비고":        stop_note,
        "1차_익절가":       _tick(tp1),
        "1차_비고":         tp1_note,
        "안전_예약매도가":  _tick(safe),
        "안전_비고":        safe_note,
    }


# ──────────────────────────────────────────────
# 2. 현재 보유 포트폴리오 로드
# ──────────────────────────────────────────────
def load_portfolio() -> pd.DataFrame:
    if not os.path.exists(PORTFOLIO_FILE):
        return pd.DataFrame()
    df = pd.read_csv(PORTFOLIO_FILE, dtype={"ticker": str})
    df["ticker"] = df["ticker"].str.zfill(6)
    return df


# ──────────────────────────────────────────────
# 3. 리포트 생성
# ──────────────────────────────────────────────
def build_report(sig: pd.DataFrame, port: pd.DataFrame, meta: dict) -> str:
    today = date.today().strftime("%Y-%m-%d")
    cv    = meta.get("cv_accuracy_mean", 0) * 100
    cv_std = meta.get("cv_accuracy_std", 0) * 100

    warn = ""
    if cv < 60:
        warn = f"\n> ⚠️ **모델 경고: CV 정확도 {cv:.1f}% — 60% 미만. 소액·관망 권장**\n"

    # 인버스 ETF 분리: 일반 매수/매도 섹션에서 제외하고 별도 헤지 섹션으로 표시
    sig_hedge = sig[sig["ticker"].isin(INVERSE_ETF_TICKERS)].copy()
    sig_main  = sig[~sig["ticker"].isin(INVERSE_ETF_TICKERS)].copy()

    lines = []
    lines.append("=" * 55)
    lines.append("📊 데일리 매매 포트폴리오 리포트")
    lines.append(f"날짜: {today}")
    lines.append(f"모델 CV 정확도: {cv:.1f}% ± {cv_std:.1f}%")
    lines.append("=" * 55)
    if warn:
        lines.append(warn)

    # ── 섹션 1: 예측 신호 요약 ──────────────────
    lines.append("\n## 1. 예측 신호 요약\n")
    lines.append("| 종목코드 | 종목명 | 신호 | 신뢰도 | 종가 | RSI | BB위치 |")
    lines.append("|---------|-------|------|--------|------|-----|--------|")
    for _, r in sig_main.iterrows():
        lines.append(
            f"| {r['ticker']} | {r['company']} | {r['신호']} | {r['신뢰도']:.1f}% "
            f"| {r['종가']:,} | {r['RSI']} | {r['BB위치']} |"
        )

    # ── 섹션 2: 현재 보유 종목 현황 ─────────────
    lines.append("\n## 2. 현재 보유 종목 현황\n")
    if port.empty:
        lines.append("보유 종목 없음 (update_portfolio.py 로 입력하세요)\n")
    else:
        lines.append("| 종목코드 | 종목명 | 수량 | 평균매수가 | 현재가 | 평가손익 | 수익률 | 오늘 신호 |")
        lines.append("|---------|-------|------|----------|--------|---------|--------|---------|")
        total_cost = total_val = 0.0
        for _, p in port.iterrows():
            cost  = float(p["avg_price"]) * float(p["quantity"])
            val   = float(p["last_price"]) * float(p["quantity"])
            pnl   = val - cost
            pct   = pnl / cost * 100 if cost else 0.0
            total_cost += cost
            total_val  += val
            s = "+" if pnl >= 0 else ""
            # 오늘 신호 찾기
            sig_row = sig[sig["ticker"] == str(p["ticker"])]
            signal_str = (
                f"{sig_row.iloc[0]['신호']} {sig_row.iloc[0]['신뢰도']}%"
                if not sig_row.empty else "-"
            )
            lines.append(
                f"| {p['ticker']} | {p['company']} | {int(p['quantity'])}주 "
                f"| {int(p['avg_price']):,} | {int(p['last_price']):,} "
                f"| {s}{int(pnl):,} | {s}{pct:.2f}% | {signal_str} |"
            )
        total_pnl = total_val - total_cost
        total_pct = total_pnl / total_cost * 100 if total_cost else 0.0
        s = "+" if total_pnl >= 0 else ""
        lines.append(f"| **합계** | | | {int(total_cost):,} | {int(total_val):,} "
                     f"| {s}{int(total_pnl):,} | {s}{total_pct:.2f}% | |")

    # ── 섹션 2-B: 예약매도 설정 가이드 ─────────
    lines.append("\n## 2-B. 예약매도 설정 가이드\n")
    if port.empty:
        lines.append("보유 종목 없음\n")
    else:
        lines.append(
            "| 종목코드 | 종목명 | 현재가 | 손절가 | 1차 익절가 | "
            "**안전 예약매도가** | 전략 근거 |"
        )
        lines.append(
            "|---------|-------|--------|--------|-----------|"
            "-----------------|---------|"
        )
        for _, p in port.iterrows():
            t = str(p["ticker"])
            sr_rows = sig[sig["ticker"] == t]
            if sr_rows.empty:
                continue
            sr  = sr_rows.iloc[0]
            tgt = calc_sell_targets(
                avg_price  = float(p["avg_price"]),
                last_price = float(p["last_price"]),
                rsi        = float(sr["RSI"]),
                bb_upper   = float(sr.get("bb_upper", 0)),
                atr14      = float(sr.get("atr14", 0)),
            )
            stop_pct = (tgt["손절가"] - float(p["avg_price"])) / float(p["avg_price"]) * 100
            safe_pct = (tgt["안전_예약매도가"] - float(p["last_price"])) / float(p["last_price"]) * 100
            lines.append(
                f"| {t} | {p['company']} "
                f"| {int(p['last_price']):,} "
                f"| {tgt['손절가']:,} ({stop_pct:+.1f}%) "
                f"| {tgt['1차_익절가']:,} "
                f"| **{tgt['안전_예약매도가']:,}** ({safe_pct:+.1f}%) "
                f"| {tgt['안전_비고']} |"
            )
        lines.append("")
        lines.append("> **손절가**: 해당 가격 이하 시 손절 — MTS 손절 주문 미리 설정 권장")
        lines.append("> **안전 예약매도가**: 오늘 MTS 지정가 매도 예약 추천가 (시장가 대비 안전한 익절)")
        lines.append("> 장중 급등 시 안전 예약매도가보다 높게 조정 가능\n")

    # ── 섹션 3: 매도 / 청산 권고 ────────────────
    lines.append("\n## 3. 매도 / 청산 권고\n")

    # 보유 종목 중 매도 신호 60% 이상
    sell_recs = []
    if not port.empty:
        for _, p in port.iterrows():
            t = str(p["ticker"])
            sig_row = sig[(sig["ticker"] == t) & (sig["신호"] == "매도") & (sig["신뢰도"] >= 60)]
            if not sig_row.empty:
                cost  = float(p["avg_price"]) * float(p["quantity"])
                val   = float(p["last_price"]) * float(p["quantity"])
                pnl   = val - cost
                pct   = pnl / cost * 100 if cost else 0.0
                s = "+" if pnl >= 0 else ""
                sell_recs.append(
                    f"| {t} | {p['company']} | {sig_row.iloc[0]['신뢰도']}% "
                    f"| {int(p['last_price']):,} | 매도 신호 (신뢰도 {sig_row.iloc[0]['신뢰도']:.1f}%), "
                    f"보유손익 {s}{pct:.1f}% |"
                )

    # 미보유 종목 중 신뢰도 60% 이상 매도 신호 (인버스 ETF 제외)
    held_tickers = set(port["ticker"].astype(str)) if not port.empty else set()
    for _, r in sig_main[(sig_main["신호"] == "매도") & (sig_main["신뢰도"] >= 60)].iterrows():
        if r["ticker"] not in held_tickers:
            sell_recs.append(
                f"| {r['ticker']} | {r['company']} | {r['신뢰도']}% "
                f"| {r['종가']:,} | 매도 신호 (미보유 — 공매도 미권장) |"
            )

    if sell_recs:
        lines.append("| 종목코드 | 종목명 | 신뢰도 | 현재가 | 매도 권고 이유 |")
        lines.append("|---------|-------|--------|--------|--------------|")
        lines.extend(sell_recs)
    else:
        lines.append("해당 없음")

    # ── 섹션 4: 매수 추천 포트폴리오 ────────────
    lines.append("\n## 4. 매수 추천 포트폴리오\n")
    lines.append(f"> 신뢰도 60% 이상 매수 신호 종목 | 총 투자가능 {int(MAX_INVEST):,}원 한도\n")

    buy_candidates = sig_main[(sig_main["신호"] == "매수") & (sig_main["신뢰도"] >= 60)].sort_values(
        "신뢰도", ascending=False
    )

    # 이미 보유 중인 종목은 제외 (이미 보유 처리)
    buy_candidates = buy_candidates[~buy_candidates["ticker"].isin(held_tickers)]

    if buy_candidates.empty:
        lines.append("신뢰도 60% 이상 매수 신호 종목 없음\n")
    else:
        lines.append("| 종목코드 | 종목명 | 추천 투자금 | 예상 매수가 | 예상 수량 | 비중 | 근거 |")
        lines.append("|---------|-------|------------|------------|---------|------|------|")
        remaining = MAX_INVEST
        subtotal  = 0
        for _, r in buy_candidates.iterrows():
            if remaining <= 0:
                break
            alloc = min(MAX_SINGLE, remaining)
            alloc = int(alloc / 10_000) * 10_000   # 만 원 단위 반올림
            qty   = max(1, int(alloc / r["종가"]))
            actual = r["종가"] * qty
            pct   = actual / TOTAL_CASH * 100
            subtotal  += actual
            remaining -= actual
            lines.append(
                f"| {r['ticker']} | {r['company']} | {int(alloc):,}원 "
                f"| {r['종가']:,}원 | {qty}주 | {pct:.1f}% "
                f"| 신뢰도 {r['신뢰도']}%, RSI {r['RSI']}, BB {r['BB위치']} |"
            )
        lines.append(f"\n소계: {int(subtotal):,}원 ({subtotal/TOTAL_CASH*100:.1f}%)")

    # ── 섹션 4-B: 관심 매수 종목 (신뢰도 50~60%) ──
    lines.append("\n## 4-B. 관심 매수 종목 (신뢰도 50~60%)\n")
    lines.append("> 60% 기준 미달이나 매수 신호 존재 — 소규모(MAX의 50%) 또는 분할 진입 참고용\n")

    watch_buy = sig_main[
        (sig_main["신호"] == "매수") &
        (sig_main["신뢰도"] >= 50) &
        (sig_main["신뢰도"] < 60)
    ].sort_values("신뢰도", ascending=False)
    watch_buy = watch_buy[~watch_buy["ticker"].isin(held_tickers)]

    if watch_buy.empty:
        lines.append("신뢰도 50~60% 매수 신호 종목 없음\n")
    else:
        lines.append("| 종목코드 | 종목명 | 신뢰도 | 종가 | RSI | BB위치 | 참고 투자금 | 참고 수량 |")
        lines.append("|---------|-------|--------|------|-----|--------|------------|---------|")
        for _, r in watch_buy.iterrows():
            alloc = int(min(MAX_SINGLE // 2, MAX_INVEST // 4) / 10_000) * 10_000
            qty   = max(1, int(alloc / r["종가"]))
            lines.append(
                f"| {r['ticker']} | {r['company']} | {r['신뢰도']}% "
                f"| {r['종가']:,} | {r['RSI']} | {r['BB위치']} "
                f"| {alloc:,}원 | {qty}주 |"
            )

    # ── 섹션 5: 관망 종목 ────────────────────────
    lines.append("\n## 5. 관망 종목\n")
    watch = sig_main[
        (sig_main["신호"] == "관망") |
        ((sig_main["신뢰도"] < 60) & (sig_main["신호"] != "관망"))
    ]
    if watch.empty:
        lines.append("해당 없음")
    else:
        lines.append("| 종목코드 | 종목명 | 신호 | 신뢰도 | 관망 이유 |")
        lines.append("|---------|-------|------|--------|---------|")
        for _, r in watch.iterrows():
            reason = "관망 신호" if r["신호"] == "관망" else f"신호 있으나 신뢰도 {r['신뢰도']}% 미달"
            lines.append(
                f"| {r['ticker']} | {r['company']} | {r['신호']} | {r['신뢰도']}% | {reason} |"
            )

    # ── 섹션 5-B: 인버스 ETF 헤지 참고 ─────────────
    lines.append("\n## 5-B. 인버스 ETF 헤지 참고\n")
    lines.append("> 시장 하락 헤지 목적 ETF — 매수 신호 = **하락 헤지 진입 고려**, 매도 신호 = **헤지 청산 고려**")
    lines.append("> 일반 종목 매수/매도 로직과 반대 방향임에 주의. 학습 데이터에서는 제외됨.\n")
    if sig_hedge.empty:
        lines.append("데이터 없음\n")
    else:
        lines.append("| 종목코드 | 종목명 | 신호 | 신뢰도 | 종가 | RSI | BB위치 | 해석 |")
        lines.append("|---------|-------|------|--------|------|-----|--------|------|")
        for _, r in sig_hedge.iterrows():
            # 인버스 ETF: 매수=하락장 대비 헤지 진입, 매도=헤지 청산
            if r["신호"] == "매수":
                interp = "하락장 헤지 진입 고려"
            elif r["신호"] == "매도":
                interp = "헤지 청산 고려 (반등 가능)"
            else:
                interp = "관망"
            lines.append(
                f"| {r['ticker']} | {r['company']} | {r['신호']} | {r['신뢰도']:.1f}% "
                f"| {r['종가']:,} | {r['RSI']} | {r['BB위치']} | {interp} |"
            )

    # ── 섹션 6: 리스크 체크 ──────────────────────
    lines.append("\n## 6. 리스크 체크\n")
    buy_cnt  = len(sig_main[sig_main["신호"] == "매수"])
    rsi_hot  = sig_main[sig_main["RSI"] > 70][["ticker", "company", "RSI"]].apply(
        lambda r: f"{r['ticker']}({r['company']}) RSI {r['RSI']}", axis=1
    ).tolist()
    rsi_cold = sig_main[sig_main["RSI"] < 30][["ticker", "company", "RSI"]].apply(
        lambda r: f"{r['ticker']}({r['company']}) RSI {r['RSI']}", axis=1
    ).tolist()

    lines.append(f"- 전체 매수 신호 종목 수: {buy_cnt}개")
    if cv < 60:
        lines.append(f"- **모델 주의사항: CV 정확도 {cv:.1f}% — 신호 신뢰도 낮음, 소액 또는 관망 권장**")
    else:
        lines.append(f"- 모델 CV 정확도: {cv:.1f}% (양호)")
    lines.append(f"- RSI 과열 종목 (RSI > 70): {', '.join(rsi_hot) if rsi_hot else '없음'}")
    lines.append(f"- RSI 과매도 종목 (RSI < 30): {', '.join(rsi_cold) if rsi_cold else '없음'}")

    # ── 섹션 7: 실행 체크리스트 ──────────────────
    lines.append("\n## 7. 실행 체크리스트\n")
    lines.append("오전 9시 장 시작 전:")
    lines.append("- [ ] 매수 종목 주문 준비 (MTS 미리 열기)")
    lines.append("- [ ] 손절가 설정: 매수가 대비 -3%")
    lines.append("- [ ] 목표가 설정: 매수가 대비 +2~3%")
    lines.append("- [ ] 장 중 급등락 시 신호와 무관하게 본인 판단 우선")

    lines.append("\n---")
    lines.append(f"*자동 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("=" * 55)

    return "\n".join(lines)


# ──────────────────────────────────────────────
# 4. 메인
# ──────────────────────────────────────────────
def main():
    print("=" * 50)
    print("아침 포트폴리오 리포트 생성")
    print("=" * 50)

    # 모델 메타 로드
    meta_path = os.path.join(MODEL_DIR, "model_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    print(f"[모델] 훈련일: {meta['trained_at']} | CV: {meta['cv_accuracy_mean']*100:.1f}%")

    print("[예측] 오늘 신호 생성 중...")
    sig  = run_prediction()
    print(f"  완료: {len(sig)}종목")

    print("[포트폴리오] 보유 현황 로드...")
    port = load_portfolio()
    print(f"  완료: {len(port)}종목 보유 중")

    report = build_report(sig, port, meta)

    # 저장
    today_str  = date.today().strftime("%Y%m%d")
    out_path   = os.path.join(DATA_DIR, f"portfolio_report_{today_str}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n[완료] 리포트 저장: {out_path}")


if __name__ == "__main__":
    main()

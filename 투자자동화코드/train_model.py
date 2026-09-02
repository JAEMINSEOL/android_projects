"""
train_model.py — XGBoost 모델 훈련 및 저장 스크립트

실행 방법:
    python train_model.py

사전 조건:
    collect_data.py 를 먼저 실행해 data/features.csv 가 있어야 합니다.

출력 (models/ 디렉터리):
    models/
    ├── xgb_model.json          — XGBoost 모델 (표준 JSON 포맷)
    ├── feature_columns.json    — 훈련에 사용된 피처 목록
    ├── scaler.pkl              — StandardScaler (피처 정규화용)
    ├── label_encoder.pkl       — LabelEncoder
    └── model_meta.json         — 훈련 메타데이터 (날짜, 성능 등)
"""

import os, json, joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

from config import DATA_DIR, MODEL_DIR

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 학습 설정
# ─────────────────────────────────────────────
# 인버스 ETF: 시장과 반대 방향 → 일반 주식과 동일 모델 학습 시 라벨 오염
EXCLUDE_TICKERS = ['114800', '252670']  # KODEX 인버스, KODEX 200선물인버스2X

# 임계값 1%→1.5%: 일상적 노이즈 수준 변동(±1%)도 매수/매도로 분류되던 문제 해소
LABEL_THRESHOLD = 0.015  # 기존 0.01

# ─────────────────────────────────────────────
# 피처 목록 (collect_data.py 와 반드시 일치)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "ma5", "ma20", "ma60",
    "rsi14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_width",
    "atr14",
    "vol_ma20", "vol_ratio",
    "close_ma5_ratio", "close_ma20_ratio", "close_ma60_ratio",
    "ma5_ma20_ratio",
    "ret_1d", "ret_3d", "ret_5d",
]

# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    feat_path = os.path.join(DATA_DIR, "features.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"{feat_path} 가 없습니다. 먼저 collect_data.py 를 실행하세요."
        )
    df = pd.read_csv(feat_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[데이터] {len(df)}행 로드 완료 | 기간: {df['date'].min().date()} ~ {df['date'].max().date()}")

    # 인버스 ETF 제외 (시장과 반대 방향 → 라벨 오염)
    before = len(df)
    ticker_col = df['ticker'].astype(str).str.lstrip('0').str.zfill(1)
    exclude_padded = [t.lstrip('0') or '0' for t in EXCLUDE_TICKERS]
    mask = df['ticker'].astype(str).apply(
        lambda t: t.lstrip('0') in exclude_padded or t in EXCLUDE_TICKERS
    )
    df = df[~mask].reset_index(drop=True)
    print(f"[필터] 인버스 ETF {before - len(df)}행 제외 (114800, 252670)")

    # 임계값 1.5%로 라벨 재계산 (collect_data의 1% 임계값 덮어쓰기)
    if 'next_ret' in df.columns:
        df['label'] = df['next_ret'].apply(
            lambda r: 2 if r >= LABEL_THRESHOLD else (0 if r <= -LABEL_THRESHOLD else 1)
            if pd.notna(r) else np.nan
        )
        print(f"[라벨] 임계값 {LABEL_THRESHOLD*100:.1f}% 재계산 완료")
        label_dist = df.dropna(subset=['label'])['label'].value_counts().sort_index()
        print(f"  매도(0)={label_dist.get(0,0)}, 관망(1)={label_dist.get(1,0)}, 매수(2)={label_dist.get(2,0)}")

    return df


# ─────────────────────────────────────────────
# 2. 전처리
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"피처 누락: {missing}")

    # 라벨 NaN 행 제거 (next_ret=NaN인 예측용 최신 row)
    before = len(df)
    df = df.dropna(subset=["label"]).copy()
    print(f"[전처리] 라벨 NaN {before - len(df)}행 제거 (예측용 최신 행) → {len(df)}행 학습")

    X = df[FEATURE_COLS].copy()
    y = df["label"].astype(int)

    # 무한값·결측 처리
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, scaler


# ─────────────────────────────────────────────
# 3. 시계열 교차검증 + 최종 훈련
# ─────────────────────────────────────────────
def train(X: np.ndarray, y: np.ndarray):
    """
    TimeSeriesSplit: 시계열 데이터 특성을 지켜 미래 정보 누수 방지
    5-fold → 각 fold 성능 확인 후 전체로 최종 훈련
    """
    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []

    params = {
        "objective":        "multi:softprob",
        "num_class":        3,           # 0:매도, 1:관망, 2:매수
        "n_estimators":     300,
        "max_depth":        5,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma":            0.1,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "random_state":     42,
        "n_jobs":           -1,
        "eval_metric":      "mlogloss",
    }

    print("\n[교차검증] TimeSeriesSplit 5-fold 시작")
    print("-" * 50)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(**params, verbosity=0)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_val)
        acc   = accuracy_score(y_val, preds)
        fold_scores.append(acc)
        print(f"  Fold {fold}: 정확도 {acc:.4f}")

    mean_acc = np.mean(fold_scores)
    std_acc  = np.std(fold_scores)
    print(f"\n  평균 정확도: {mean_acc:.4f} ± {std_acc:.4f}")

    # 전체 데이터로 최종 모델 훈련
    print("\n[훈련] 전체 데이터로 최종 모델 훈련 중...")
    final_model = xgb.XGBClassifier(**params, verbosity=0)
    final_model.fit(X, y)

    return final_model, fold_scores


# ─────────────────────────────────────────────
# 4. 피처 중요도 출력
# ─────────────────────────────────────────────
def print_feature_importance(model: xgb.XGBClassifier):
    importance = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    print("\n[피처 중요도] 상위 10개")
    print("-" * 35)
    for _, row in fi_df.head(10).iterrows():
        bar = "#" * int(row["importance"] * 200)
        print(f"  {row['feature']:25s} {row['importance']:.4f} {bar}")

    return fi_df


# ─────────────────────────────────────────────
# 5. 모델 저장
# ─────────────────────────────────────────────
def save_model(model, scaler, fold_scores, fi_df):
    # 5-1. XGBoost 모델 (JSON — 버전 독립적 표준 포맷)
    model_path = os.path.join(MODEL_DIR, "xgb_model.json")
    model.save_model(model_path)

    # 5-2. 피처 컬럼 목록
    feat_path = os.path.join(MODEL_DIR, "feature_columns.json")
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLS, f, ensure_ascii=False, indent=2)

    # 5-3. Scaler (sklearn joblib)
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # 5-4. 피처 중요도
    fi_path = os.path.join(MODEL_DIR, "feature_importance.csv")
    fi_df.to_csv(fi_path, index=False, encoding="utf-8-sig")

    # 5-5. 메타데이터
    meta = {
        "trained_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_features":     len(FEATURE_COLS),
        "cv_accuracy_mean": float(np.mean(fold_scores)),
        "cv_accuracy_std":  float(np.std(fold_scores)),
        "cv_fold_scores":   [float(s) for s in fold_scores],
        "classes":        {"0": "매도", "1": "관망", "2": "매수"},
        "threshold":      {"매수": f"+{LABEL_THRESHOLD*100:.1f}%", "매도": f"-{LABEL_THRESHOLD*100:.1f}%"},
        "excluded_tickers": EXCLUDE_TICKERS,
        "files": {
            "model":    "xgb_model.json",
            "features": "feature_columns.json",
            "scaler":   "scaler.pkl",
            "fi":       "feature_importance.csv",
        }
    }
    meta_path = os.path.join(MODEL_DIR, "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[저장 완료] {MODEL_DIR}/")
    print(f"  xgb_model.json          - XGBoost 모델")
    print(f"  feature_columns.json    - 피처 목록")
    print(f"  scaler.pkl              - 정규화 스케일러")
    print(f"  feature_importance.csv  - 피처 중요도")
    print(f"  model_meta.json         - 훈련 메타데이터")


# ─────────────────────────────────────────────
# 6. 실행
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("XGBoost 모델 훈련 시작")
    print("=" * 50)

    df              = load_data()
    X, y, scaler    = preprocess(df)
    model, scores   = train(X, y)
    fi_df           = print_feature_importance(model)

    # 전체 데이터 최종 성능 리포트
    preds = model.predict(X)
    print("\n[최종 모델 전체 데이터 성능]")
    print(classification_report(y, preds, target_names=["매도(0)", "관망(1)", "매수(2)"]))

    save_model(model, scaler, scores, fi_df)
    print("\n훈련 완료!")


if __name__ == "__main__":
    main()

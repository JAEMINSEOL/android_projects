"""모델 예측 이탈 원인 분석"""
import pandas as pd, numpy as np, json, pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

with open('models/feature_columns.json', encoding='utf-8') as f:
    FEAT_COLS = json.load(f)
model = xgb.XGBClassifier()
model.load_model('models/xgb_model.json')

feat = pd.read_csv('data/features.csv')
feat['date'] = pd.to_datetime(feat['date'])
feat_kr = feat[feat['asset_type'] != 'us_stocks'].copy()
valid = feat_kr.dropna(subset=['next_ret'] + FEAT_COLS).copy().reset_index(drop=True)

# scaler.pkl 버전 불일치 시 현재 데이터로 재피팅
try:
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X = scaler.transform(valid[FEAT_COLS])
except Exception:
    scaler = StandardScaler()
    X = scaler.fit_transform(valid[FEAT_COLS])
    print("[경고] scaler.pkl 로드 실패 - 현재 데이터로 재피팅")
proba = model.predict_proba(X)
valid['pred'] = np.argmax(proba, axis=1)
valid['confidence'] = np.max(proba, axis=1)
valid['correct'] = (valid['pred'] == valid['label']).astype(int)

print("=" * 55)
print("1. 신뢰도 구간별 정확도")
print("=" * 55)
bins  = [0, 0.40, 0.50, 0.60, 0.70, 1.0]
lbls  = ['<40%','40-50%','50-60%','60-70%','>70%']
valid['conf_bin'] = pd.cut(valid['confidence'], bins=bins, labels=lbls)
res = valid.groupby('conf_bin', observed=True).agg(
    n=('correct','count'),
    accuracy=('correct','mean')
).round(3)
print(res)

print()
print("=" * 55)
print("2. 매수 예측 종목의 실제 다음날 수익률 분포")
print("=" * 55)
for pred_label, name in [(0,'매도예측'),(1,'관망예측'),(2,'매수예측')]:
    s = valid[valid['pred'] == pred_label]['next_ret']
    print(f"{name}: n={len(s):4d}, 평균={s.mean()*100:+.2f}%, "
          f"중앙값={s.median()*100:+.2f}%, 양수={( s>0).mean()*100:.1f}%")

print()
print("=" * 55)
print("3. 인버스 ETF 포함 여부가 모델에 미치는 영향")
print("=" * 55)
inverse_tickers = ['114800', '252670']  # KODEX 인버스, 200선물인버스2X
inv = feat_kr[feat_kr['ticker'].astype(str).isin(inverse_tickers)]
non_inv = feat_kr[~feat_kr['ticker'].astype(str).isin(inverse_tickers)]
print(f"인버스 ETF 행 수: {len(inv)}")
if len(inv) > 0:
    inv_v = inv.dropna(subset=['next_ret'])
    print(f"  - 매수라벨(2) 비율: {(inv_v['label']==2).mean()*100:.1f}%")
    print(f"  - 매도라벨(0) 비율: {(inv_v['label']==0).mean()*100:.1f}%")
    print(f"  - next_ret 평균: {inv_v['next_ret'].mean()*100:+.3f}%")

print()
print("=" * 55)
print("4. 자산군별 예측 정확도")
print("=" * 55)
for atype in valid['asset_type'].unique():
    sub = valid[valid['asset_type'] == atype]
    acc = sub['correct'].mean()
    n = len(sub)
    # 매수 예측 적중률
    buy_pred = sub[sub['pred'] == 2]
    buy_acc = buy_pred['correct'].mean() if len(buy_pred) > 0 else float('nan')
    print(f"{atype}: n={n}, 전체정확도={acc:.3f}, 매수예측정확도={buy_acc:.3f} (n={len(buy_pred)})")

print()
print("=" * 55)
print("5. 최근 30일 vs 전체 신호 품질 비교")
print("=" * 55)
cutoff = valid['date'].max() - pd.Timedelta(days=30)
recent = valid[valid['date'] >= cutoff]
old    = valid[valid['date'] < cutoff]
print(f"전체 기간  : n={len(old)},  정확도={old['correct'].mean():.3f}")
print(f"최근 30일  : n={len(recent)}, 정확도={recent['correct'].mean():.3f}")
if len(recent) > 0:
    for pred_label, name in [(2,'매수'),(0,'매도')]:
        sub = recent[recent['pred'] == pred_label]
        if len(sub) > 0:
            print(f"  최근 {name}예측 정확도: {sub['correct'].mean():.3f} (n={len(sub)})")

print()
print("=" * 55)
print("6. 라벨 1% 임계값 민감도 - 임계값별 라벨 분포")
print("=" * 55)
for thresh in [0.005, 0.010, 0.015, 0.020]:
    nr = feat_kr.dropna(subset=['next_ret'])['next_ret']
    buy  = (nr >=  thresh).sum()
    sell = (nr <= -thresh).sum()
    hold = len(nr) - buy - sell
    print(f"임계값 {thresh*100:.1f}%: 매수={buy}({buy/len(nr)*100:.1f}%) "
          f"관망={hold}({hold/len(nr)*100:.1f}%) 매도={sell}({sell/len(nr)*100:.1f}%)")

print()
print("=" * 55)
print("7. 피처 중요도 상위 vs 하위 그룹 기여")
print("=" * 55)
fi = pd.read_csv('models/feature_importance.csv')
top5    = fi.nlargest(5, 'importance')['feature'].tolist()
bottom5 = fi.nsmallest(5, 'importance')['feature'].tolist()
print("상위 5 피처:", top5)
print("하위 5 피처:", bottom5)
print(f"상위5 중요도 합: {fi.nlargest(5,'importance')['importance'].sum():.3f}")
print(f"피처 중요도 균등도(std): {fi['importance'].std():.4f}")
print("→ 중요도가 균등(std 작을수록 특정 피처에 과의존 없음)")

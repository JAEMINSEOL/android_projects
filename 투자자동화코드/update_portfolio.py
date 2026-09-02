"""
update_portfolio.py — 매일 저녁 보유 포트폴리오 입력/업데이트

실행:
    python update_portfolio.py

출력:
    data/my_portfolio.csv — 보유 종목 현황 (아침 리포트에서 자동 참조)
"""

import os
import pandas as pd
from datetime import date
from config import DATA_DIR

PORTFOLIO_FILE = os.path.join(DATA_DIR, "my_portfolio.csv")

TICKER_NAMES = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "005380": "현대차",
    "035420": "NAVER",
    "051910": "LG화학",
    "006400": "삼성SDI",
    "035720": "카카오",
    "000270": "기아",
    "105560": "KB금융",
    "055550": "신한지주",
}


def load_portfolio() -> pd.DataFrame:
    if os.path.exists(PORTFOLIO_FILE):
        df = pd.read_csv(PORTFOLIO_FILE, dtype={"ticker": str})
        df["ticker"] = df["ticker"].str.zfill(6)
        return df
    return pd.DataFrame(columns=[
        "ticker", "company", "quantity", "avg_price", "last_price", "last_updated"
    ])


def save_portfolio(df: pd.DataFrame):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(PORTFOLIO_FILE, index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] {PORTFOLIO_FILE}")


def print_portfolio(df: pd.DataFrame):
    if df.empty:
        print("  (보유 종목 없음)")
        return

    print(f"\n{'종목코드':>8}  {'종목명':>10}  {'수량':>5}  "
          f"{'평균매수가':>10}  {'현재가':>10}  {'평가손익':>12}  {'수익률':>8}")
    print("-" * 78)

    total_cost = total_value = 0.0
    for _, row in df.iterrows():
        cost  = float(row["avg_price"]) * float(row["quantity"])
        value = float(row["last_price"]) * float(row["quantity"])
        pnl   = value - cost
        pct   = pnl / cost * 100 if cost else 0.0
        total_cost  += cost
        total_value += value
        s = "+" if pnl >= 0 else ""
        print(
            f"  {str(row['ticker']):>8}  {str(row['company']):>10}  {int(row['quantity']):>5}  "
            f"  {int(row['avg_price']):>10,}  {int(row['last_price']):>10,}  "
            f"  {s}{int(pnl):>10,}  {s}{pct:>7.2f}%"
        )

    total_pnl = total_value - total_cost
    total_pct = total_pnl / total_cost * 100 if total_cost else 0.0
    s = "+" if total_pnl >= 0 else ""
    print("-" * 78)
    print(f"  {'합계':>45}  "
          f"{int(total_cost):>10,}  {int(total_value):>10,}  "
          f"  {s}{int(total_pnl):>10,}  {s}{total_pct:>7.2f}%")
    print(f"\n  기준일: {df['last_updated'].max()}")


def add_or_update(df: pd.DataFrame) -> pd.DataFrame:
    raw = input("  종목코드 (예: 105560): ").strip()
    ticker = raw.zfill(6)
    company = TICKER_NAMES.get(ticker, "")
    if not company:
        company = input(f"  종목명 ({ticker}): ").strip()

    qty_str       = input("  보유 수량 (주): ").strip().replace(",", "")
    avg_str       = input("  평균 매수가 (원): ").strip().replace(",", "")
    last_str      = input("  오늘 종가 (원): ").strip().replace(",", "")

    new_row = {
        "ticker":       ticker,
        "company":      company,
        "quantity":     float(qty_str),
        "avg_price":    float(avg_str),
        "last_price":   float(last_str),
        "last_updated": str(date.today()),
    }

    if ticker in df["ticker"].values:
        for k, v in new_row.items():
            df.loc[df["ticker"] == ticker, k] = v
        print(f"  [{ticker}] {company} 수정 완료")
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"  [{ticker}] {company} 추가 완료")
    return df


def update_prices_only(df: pd.DataFrame) -> pd.DataFrame:
    print("  (엔터 = 이전 값 유지)")
    for i, row in df.iterrows():
        raw = input(
            f"  {row['ticker']} {row['company']} 현재가 (현재 {int(row['last_price']):,}원): "
        ).strip().replace(",", "")
        if raw:
            df.at[i, "last_price"] = float(raw)
        df.at[i, "last_updated"] = str(date.today())
    return df


def main():
    print("=" * 55)
    print("  보유 포트폴리오 업데이트")
    print(f"  날짜: {date.today()}")
    print("=" * 55)

    df = load_portfolio()

    while True:
        print("\n[ 현재 포트폴리오 ]")
        print_portfolio(df)

        print("\n  1. 종목 추가 / 수정")
        print("  2. 종목 삭제")
        print("  3. 현재가만 일괄 업데이트")
        print("  4. 저장 후 종료")
        print("  5. 저장 없이 종료")

        choice = input("\n선택 (1-5): ").strip()

        if choice == "1":
            df = add_or_update(df)

        elif choice == "2":
            raw = input("  삭제할 종목코드: ").strip().zfill(6)
            if raw in df["ticker"].values:
                name = df.loc[df["ticker"] == raw, "company"].values[0]
                df = df[df["ticker"] != raw].reset_index(drop=True)
                print(f"  [{raw}] {name} 삭제 완료")
            else:
                print(f"  [{raw}] 보유 목록에 없습니다.")

        elif choice == "3":
            df = update_prices_only(df)

        elif choice == "4":
            save_portfolio(df)
            print("\n[ 최종 포트폴리오 ]")
            print_portfolio(df)
            print("\n내일 아침 리포트에 반영됩니다.")
            break

        elif choice == "5":
            print("  저장하지 않고 종료합니다.")
            break

        else:
            print("  1~5 중에서 선택하세요.")


if __name__ == "__main__":
    main()

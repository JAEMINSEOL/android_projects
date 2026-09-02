# -*- coding: utf-8 -*-
"""
daily_auto.py — 무인 데일리 파이프라인 (Windows 작업 스케줄러용)

수행 순서:
    1) collect_data.py 실행 (최신 시세 수집)
    2) 보유 포트폴리오 last_price 를 features.csv 최신 종가로 자동 갱신
    3) morning_report.py 실행 (리포트 생성)
    4) 생성된 리포트를 Notion '데일리 포트폴리오' DB에 새 항목으로 등록
       (제목 + 작성일 속성 포함)

수동 실행:
    python daily_auto.py

로그: data/daily_auto.log 에 append
"""

import os
import sys
import io
import json
import subprocess
import urllib.request
import urllib.error
from datetime import date, datetime

import pandas as pd

# ── 설정 ────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
PORTFOLIO = os.path.join(DATA_DIR, "my_portfolio.csv")
FEATURES  = os.path.join(DATA_DIR, "features.csv")
LOG_FILE  = os.path.join(DATA_DIR, "daily_auto.log")
PY        = sys.executable

NOTION_TOKEN = os.environ.get("NOTION_API_TOKEN", "").strip()
NOTION_DB_ID = os.environ.get("NOTION_DB_ID", "395425af-7462-80ed-a0d2-d1fbed412c32").strip()
NOTION_VER   = "2022-06-28"

# 표준출력 UTF-8 강제 (스케줄러 환경 인코딩 방어)
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass


def log(msg: str):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── 1) 서브프로세스 실행 ────────────────────────────────
def run_step(script: str):
    log(f"실행: {script}")
    r = subprocess.run(
        [PY, os.path.join(BASE_DIR, script)],
        cwd=BASE_DIR, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        log(f"[오류] {script} 종료코드 {r.returncode}")
        log((r.stderr or "")[-1500:])
        raise SystemExit(f"{script} 실패")
    log(f"완료: {script}")


# ── 2) 보유가 자동 갱신 ─────────────────────────────────
def update_portfolio_prices():
    if not os.path.exists(PORTFOLIO):
        log("보유 포트폴리오 파일 없음 — 가격 갱신 생략")
        return
    pf = pd.read_csv(PORTFOLIO, dtype={"ticker": str})
    if pf.empty:
        log("보유 종목 없음 — 가격 갱신 생략")
        return
    pf["ticker"] = pf["ticker"].str.zfill(6)

    feat = pd.read_csv(FEATURES, dtype={"ticker": str})
    feat["ticker"] = feat["ticker"].str.zfill(6)
    latest_date = feat["date"].max()

    for i, row in pf.iterrows():
        sub = feat[feat["ticker"] == row["ticker"]].sort_values("date")
        if sub.empty:
            log(f"  {row['ticker']} {row['company']}: features 에 없음 — 유지")
            continue
        close = float(sub.iloc[-1]["close"])
        pf.at[i, "last_price"]   = int(close)
        pf.at[i, "last_updated"] = latest_date
        log(f"  {row['ticker']} {row['company']}: last_price -> {int(close):,} ({latest_date})")

    pf.to_csv(PORTFOLIO, index=False, encoding="utf-8-sig")
    log("보유 포트폴리오 가격 갱신 저장 완료")


# ── 4) Notion 등록 ──────────────────────────────────────
WANTED_SECTIONS = (
    "## 2.", "## 2-B.", "## 3.", "## 4.", "## 4-B.", "## 5-B.", "## 6.",
)

def _rt(text):
    return [{"type": "text", "text": {"content": text[:1900]}}]

def _table_block(rows):
    """rows: 마크다운 표 라인 리스트 (0=헤더, 1=구분선, 2+=데이터)"""
    def cells(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]
    header = cells(rows[0])
    width  = len(header)
    def make(cs):
        cs = (cs + [""] * width)[:width]
        return {"type": "table_row",
                "table_row": {"cells": [_rt(c) for c in cs]}}
    children = [make(header)]
    for ln in rows[2:]:
        children.append(make(cells(ln)))
    return {"object": "block", "type": "table",
            "table": {"table_width": width, "has_column_header": True,
                      "has_row_header": False, "children": children}}

def md_to_blocks(md_path):
    with open(md_path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    blocks = []
    i = 0
    include = False
    while i < len(lines):
        ln = lines[i].rstrip()
        stripped = ln.strip()

        if stripped.startswith("## "):
            include = any(stripped.startswith(s) for s in WANTED_SECTIONS)
            if include:
                blocks.append({"object": "block", "type": "heading_2",
                               "heading_2": {"rich_text": _rt(stripped[3:].strip())}})
            i += 1
            continue

        if not include:
            i += 1
            continue

        if not stripped or set(stripped) <= {"="}:
            i += 1
            continue

        if stripped.startswith("|"):  # 표 수집
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i]); i += 1
            if len(tbl) >= 2:
                blocks.append(_table_block(tbl))
            continue

        if stripped.startswith(">"):
            blocks.append({"object": "block", "type": "quote",
                           "quote": {"rich_text": _rt(stripped.lstrip("> ").strip())}})
            i += 1
            continue

        if stripped.startswith("- "):
            blocks.append({"object": "block", "type": "bulleted_list_item",
                           "bulleted_list_item": {"rich_text": _rt(stripped[2:].strip())}})
            i += 1
            continue

        if stripped.startswith("---"):
            blocks.append({"object": "block", "type": "divider", "divider": {}})
            i += 1
            continue

        blocks.append({"object": "block", "type": "paragraph",
                       "paragraph": {"rich_text": _rt(stripped)}})
        i += 1

    return blocks

def _notion(url, method, body=None):
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VER,
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))

def push_to_notion(report_date: str, md_path: str):
    if not NOTION_TOKEN:
        log("[Notion] NOTION_API_TOKEN 환경변수 없음 — 등록 생략")
        return
    blocks = md_to_blocks(md_path)
    header = {"object": "block", "type": "callout",
              "callout": {"rich_text": _rt("자동 생성 리포트 (KIS API + XGBoost) — 모델 신호는 참고용, 투자 판단은 본인 책임"),
                          "icon": {"type": "emoji", "emoji": "🤖"}}}
    blocks = [header] + blocks

    payload = {
        "parent": {"database_id": NOTION_DB_ID},
        "properties": {
            "이름": {"title": _rt(f"📊 데일리 리포트 {report_date}")},
            "작성일": {"date": {"start": report_date}},
        },
        "children": blocks[:100],
    }
    try:
        res = _notion("https://api.notion.com/v1/pages", "POST", payload)
        page_id = res["id"]
        log(f"[Notion] 페이지 생성 완료: {res.get('url','')}")
        # 100개 초과 블록 추가 append
        rest = blocks[100:]
        while rest:
            chunk, rest = rest[:100], rest[100:]
            _notion(f"https://api.notion.com/v1/blocks/{page_id}/children",
                    "PATCH", {"children": chunk})
        log("[Notion] 블록 등록 완료")
    except urllib.error.HTTPError as e:
        log(f"[Notion] HTTP {e.code}: {e.read().decode('utf-8')[:800]}")
    except Exception as e:
        log(f"[Notion] 실패: {e}")


# ── main ────────────────────────────────────────────────
def main():
    log("=" * 50)
    log("데일리 자동 파이프라인 시작")

    run_step("collect_data.py")
    update_portfolio_prices()
    run_step("morning_report.py")

    today = str(date.today())
    md_path = os.path.join(DATA_DIR, f"portfolio_report_{today.replace('-', '')}.md")
    if not os.path.exists(md_path):
        # 리포트가 features 최신일 기준으로 생성될 수 있으므로 최신 리포트 탐색
        reports = sorted(
            fn for fn in os.listdir(DATA_DIR)
            if fn.startswith("portfolio_report_") and fn.endswith(".md")
        )
        if reports:
            md_path = os.path.join(DATA_DIR, reports[-1])
            today = reports[-1].replace("portfolio_report_", "").replace(".md", "")
            today = f"{today[:4]}-{today[4:6]}-{today[6:]}"
    log(f"리포트 파일: {os.path.basename(md_path)}")

    push_to_notion(today, md_path)
    log("데일리 자동 파이프라인 종료")


if __name__ == "__main__":
    main()

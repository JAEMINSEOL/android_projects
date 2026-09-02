# config_extended.py — 다자산 확장 설정
# 기존 config.py 를 이 파일로 교체하세요.


APP_KEY    = "PScPzr5wMDwzxhcmVnJiHjnCjbuna0vWEUOs"
APP_SECRET = "Ib/tZLByJAL0S5sOh083r1ILQbO8OCE5tpzcsTaGlfVYvObQ7NUSDQ3w+t/Sum2dBkQ2dWm+q3kHQyuxvwnOWANe0RmBN8uPZEOO2jLdbkHcizQTKq5dSGjpW3uCpXdscAs3qGijh8gFAaMd9YEB3kILg8cWprkDa3IYvMGLc4aGrzOZVKI="
ACCOUNT_NO = "44570307-01"   # 계좌번호 (앞8자리-뒤2자리)

# 실전투자: "https://openapi.koreainvestment.com:9443"
# 모의투자: "https://openapivts.koreainvestment.com:29443"
BASE_URL = "https://openapi.koreainvestment.com:9443"



MODEL_DIR = "./models"
DATA_DIR  = "./data"

# ── 상위 n% 종목 필터 설정 ───────────────────────────────
# 포트폴리오에 포함할 상위 종목 비율 (0.0 ~ 1.0)
# 예: 0.3 → 신호 발생 종목 중 신뢰도 상위 30%만 포함
TOP_N_PERCENT = 0.3

# 최소 신뢰도 하한 (TOP_N_PERCENT 와 AND 조건)
MIN_CONFIDENCE = 0.60

# ── 자산군별 티커 ────────────────────────────────────────
ASSETS = {

    # 코스피 대형주 (기존)
    "kospi_large": [
        "005930",  # 삼성전자
        "000660",  # SK하이닉스
        "005380",  # 현대차
        "035420",  # NAVER
        "051910",  # LG화학
        "006400",  # 삼성SDI
        "035720",  # 카카오
        "000270",  # 기아
        "105560",  # KB금융
        "055550",  # 신한지주
        "012330",  # 현대모비스
        "028260",  # 삼성물산
        "066570",  # LG전자
        "003550",  # LG
        "017670",  # SK텔레콤
        "030200",  # KT
        "096770",  # SK이노베이션
        "011200",  # HMM
        "010950",  # S-Oil
        "086790",  # 하나금융지주
    ],

    # 국내 ETF (KIS API 동일 엔드포인트 사용)
    "etf_domestic": [
        "069500",  # KODEX 200
        "229200",  # KODEX 코스닥150
        "305720",  # KODEX 2차전지산업
        "091160",  # KODEX 반도체
        "139220",  # KODEX IT
        "114800",  # KODEX 인버스 (하락 헤지용)
        "252670",  # KODEX 200선물인버스2X
        "148020",  # KODEX 헬스케어
        "364980",  # KODEX Fn수소경제테마
        "379800",  # KODEX 미국S&P500
    ],

    # KRX 금현물
    "krx_gold": [
        "4GLD",    # KRX 금현물 (4자리 시세코드)
    ],

    # 해외 주식 (yfinance 사용 — KIS API 와 별도 수집)
    "us_stocks": [
        "AAPL",   # 애플
        "MSFT",   # 마이크로소프트
        "NVDA",   # 엔비디아
        "GOOGL",  # 알파벳
        "AMZN",   # 아마존
        "META",   # 메타
        "TSLA",   # 테슬라
        "TSM",    # TSMC
        "ASML",   # ASML
        "AMD",    # AMD
    ],
}

# 하위 호환: 기존 코드에서 TICKERS 를 참조하는 부분 유지
TICKERS = (
    ASSETS["kospi_large"] +
    ASSETS["etf_domestic"] +
    ASSETS["krx_gold"]
)

# 해외 주식 전용 (yfinance 수집)
TICKERS_US = ASSETS["us_stocks"]

# 자산군 라벨 (리포트 구분용)
ASSET_LABEL = {
    "kospi_large":  "코스피 대형주",
    "etf_domestic": "국내 ETF",
    "krx_gold":     "KRX 금현물",
    "us_stocks":    "미국 주식",
}

# 해외주식 추가 매크로 피처 수집 여부
FETCH_MACRO = True  # True: USD/KRW, VIX, 나스닥 지수 함께 수집

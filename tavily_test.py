import os
import re
from enum import Enum
from urllib.parse import urlparse

from tavily import TavilyClient

client = TavilyClient(
    api_key="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"
)

product = {
    "id": 343,
    "title": "미쏘 basic 쏘쿨 핏업 와이드핏 데님 팬츠",
    "link": "https://search.shopping.naver.com/catalog/59809399296",
    "lprice": 51300,
    "brand": "미쏘",
    "collected_at": "2026-07-01T06:22:23.260478",
}

EXCLUDED_DOMAINS = {
    "m.mixxo.com",
}


# =========================================================
# 검색
# =========================================================


def search_latest_price(product: dict):
    title = product["title"]
    brand = product["brand"]

    # URL 자체를 검색어로 강제하는 것은 별 도움 안 될 가능성이 큼
    query = f'"{title}" "{brand}" 가격 할인판매가'

    print("검색 Query:", query)

    result = client.search(
        query=query,
        search_depth="advanced",
        max_results=10,
    )

    return result.get("results", [])


def is_excluded_url(url: str) -> bool:
    try:
        domain = urlparse(url).netloc.lower()

        return domain in EXCLUDED_DOMAINS
    except Exception:
        return True


# =========================================================
# 가격
# =========================================================


def extract_money(text: str) -> list[int]:
    matches = re.findall(
        r"(?<!\d)(\d{1,3}(?:,\d{3})+)(?:\s*원)?",
        text,
    )

    return [int(x.replace(",", "")) for x in matches]


def extract_sale_price(text: str) -> int | None:
    """
    우선순위

    1. 할인판매가
    2. 판매가

    반드시 같은 line 안에 있는 가격만 사용하여
    배송비/적립금/다른 상품 가격이 섞이는 것을 방지.
    """

    if not text:
        return None

    # Tavily markdown escape 제거
    text = text.replace("\\_", "_")

    lines = text.splitlines()

    # -----------------------------------------------------
    # 1. 할인판매가 우선
    #
    # | 할인판매가 | 34,930 |
    # -----------------------------------------------------

    for line in lines:
        if "할인판매가" in line or "할인 판매가" in line:
            prices = extract_money(line)

            if prices:
                print("  [가격근거/할인판매가]", line.strip())
                return min(prices)

    # -----------------------------------------------------
    # 2. 판매가
    #
    # | 판매가 | 49,900 |
    #
    # 또는
    #
    # 판매가 49,900원 15,900원 68%
    #
    # 같은 line이라면 할인된 작은 값을 사용
    # -----------------------------------------------------

    for line in lines:
        # "소비자가", "할인판매가" 제외
        if "판매가" not in line:
            continue

        if "소비자가" in line or "할인판매가" in line:
            continue

        prices = extract_money(line)

        if prices:
            print("  [가격근거/판매가]", line.strip())
            return min(prices)

    return None


# =========================================================
# 재고
# =========================================================


class StockStatus(Enum):
    SOLD_OUT = "sold_out"
    IN_STOCK = "in_stock"
    UNKNOWN = "unknown"


def check_stock_status(text: str) -> StockStatus:
    if not text:
        return StockStatus.UNKNOWN

    # 공백 / 줄바꿈 제거
    compact = re.sub(r"\s+", "", text).lower()

    # -----------------------------------------------------
    # 확실한 품절 문구만 사용
    # -----------------------------------------------------

    sold_out_patterns = [
        "상품이품절되었습니다",
        "현재품절되었습니다",
        "현재품절",
        "일시품절",
        "재고가없습니다",
        "재고없음",
        "재고소진",
        "판매종료",
        "판매중단",
        "구매불가",
        "soldout",
        "outofstock",
    ]

    for pattern in sold_out_patterns:
        if pattern in compact:
            return StockStatus.SOLD_OUT

    # -----------------------------------------------------
    # 명시적인 판매 가능 신호
    #
    # 주의:
    # 쇼핑몰 HTML에는 구매하기 템플릿이 항상 들어가는 경우가 있어
    # "구매하기" 하나만으로 IN_STOCK 판단하면 안 됨.
    # -----------------------------------------------------

    in_stock_patterns = [
        "현재구매가능",
        "재고있음",
        "instock",
    ]

    for pattern in in_stock_patterns:
        if pattern in compact:
            return StockStatus.IN_STOCK

    return StockStatus.UNKNOWN


# =========================================================
# 상품 코드
# =========================================================


def extract_product_codes(text: str) -> list[str]:
    """
    MIWTJG560J 같은 상품코드를 찾아냄.
    """

    if not text:
        return []

    matches = re.findall(
        r"\b[A-Z]{5,}\d+[A-Z0-9]*\b",
        text.upper(),
    )

    # 중복 제거
    return list(dict.fromkeys(matches))


def is_same_product(product: dict, title: str, content: str) -> bool:
    """
    product_code가 DB에 있다면 SKU exact match를 강제.

    product_code가 없다면 현재는 True.
    → 정확한 상품 판별이 필요하면 DB에 SKU 저장 권장.
    """

    target_code = product.get("product_code")

    if not target_code:
        return True

    text = f"{title}\n{content}".upper()

    return target_code.upper() in text


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    results = search_latest_price(product)

    candidates = []

    for idx, item in enumerate(results, 1):
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")

        text = f"{title}\n{content}"

        if is_excluded_url(url):
            print(f"[제외] 비정상/모바일 URL: {url}")
            continue

        print("\n" + "=" * 80)
        print(f"[{idx}] {title}")
        print("URL:", url)

        # -------------------------------------------------
        # 검색된 상품 코드 확인
        # -------------------------------------------------

        product_codes = extract_product_codes(text)

        print("발견 SKU:", product_codes)

        # -------------------------------------------------
        # 동일 상품 검사
        # -------------------------------------------------

        if not is_same_product(
            product,
            title,
            content,
        ):
            print("[제외] SKU 불일치")
            continue

        # -------------------------------------------------
        # 재고 판정
        # -------------------------------------------------

        stock_status = check_stock_status(text)

        print("재고 상태:", stock_status.value)

        if stock_status == StockStatus.SOLD_OUT:
            print("[제외] 품절 상품")
            continue

        # -------------------------------------------------
        # 가격 추출
        # -------------------------------------------------

        price = extract_sale_price(text)

        print("발견 가격:", price)

        if price is None:
            print("[제외] 가격 추출 실패")
            continue

        # -------------------------------------------------
        # 후보 추가
        # -------------------------------------------------

        candidates.append(
            {
                "title": title,
                "url": url,
                "price": price,
                "stock_status": stock_status.value,
                "product_codes": product_codes,
            }
        )

    # =====================================================
    # 모든 검색 결과 순회 후 최저가 선택
    # =====================================================

    print("\n")
    print("=" * 80)

    if candidates:
        lowest = min(
            candidates,
            key=lambda x: x["price"],
        )

        print("최저 가격 후보")
        print("-" * 80)

        print("가격 :", lowest["price"])
        print("제목 :", lowest["title"])
        print("URL  :", lowest["url"])
        print("재고 :", lowest["stock_status"])
        print("SKU  :", lowest["product_codes"])

    else:
        print("가격 후보를 찾지 못했습니다.")

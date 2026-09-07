import os
import re
from enum import Enum
from urllib.parse import urlparse

from tavily import TavilyClient

client = TavilyClient(
    api_key="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"
)

product = {
    # "id": 343,
    "title": "나이키 에어맥스 모토 2K 신발 IO9279",
    "brand": "나이키",
    "cate": "남성신발",
    "sale": "네이버",
    "collected_at": "2026-07-01T06:22:23.260478",
}

EXCLUDED_DOMAINS = {"m.mixxo.com", "search.shopping.naver.com", "www.11st.co.kr"}


# =========================================================
# 검색
# =========================================================


def search_latest_price(product: dict):
    title = product["title"]
    brand = product["brand"]
    sale = product["sale"]
    cate = product["cate"]
    # URL 자체를 검색어로 강제하는 것은 별 도움 안 될 가능성이 큼
    # query = f' "{sale}" 쇼핑몰에서 판매하는 "{cate}" 카테고리 "{title}"의 판매가 찾아'
    query = '"장원영" "의류" "종류" 가격'
    print("검색 Query:", query)

    result = client.search(
        query=query,
        search_depth="advanced",
        max_results=3,
        chunks_per_source=3,
        exclude_domains=["tistory.com", "blog.naver.com", "www.instagram.com"],
    )

    return result.get("results", [])


# =========================================================
# 가격
# =========================================================


def extract_money(text: str) -> list[int]:
    matches = re.findall(
        r"(?<!\d)(\d{1,3}(?:,\d{3})+|\d+)\s*원?",
        text,
    )

    return [int(x.replace(",", "")) for x in matches]


def extract_sale_price(text: str) -> int | None:
    if not text:
        return None

    text = text.replace("\\_", "_")

    lines = text.splitlines()

    money = r"(\d{1,3}(?:,\d{3})+|\d+)"

    # =====================================================
    # 0. 할인율 바로 뒤 가격
    # 30% 19,500원
    # =====================================================
    match = re.search(
        rf"\d{{1,3}}%\s*{money}\s*원",
        text,
    )

    match = re.search(
        r"판매가\s*[:：]?\s*(\d{1,3}(?:,\d{3})+|\d+)\s*원",
        text,
    )

    if match:
        price = int(match.group(1).replace(",", ""))
        print("  [가격근거/할인율 판매가]", match.group(0))
        return price

    # =====================================================
    # 1. 할인판매가
    # =====================================================
    for line in lines:
        if "할인판매가" not in line and "할인 판매가" not in line:
            continue

        before = re.search(
            rf"{money}\s*원?\s*(?:할인판매가|할인\s*판매가)",
            line,
        )

        if before:
            return int(before.group(1).replace(",", ""))

        after = re.search(
            rf"(?:할인판매가|할인\s*판매가)"
            rf"[\s:：|・\-]*"
            rf"{money}\s*원?",
            line,
        )

        if after:
            return int(after.group(1).replace(",", ""))

    # =====================================================
    # 2. 판매가
    # =====================================================
    for line in lines:
        if "판매가" not in line:
            continue

        if "소비자가" in line or "할인판매가" in line:
            continue

        before = re.search(
            rf"{money}\s*원?\s*판매가",
            line,
        )

        if before:
            return int(before.group(1).replace(",", ""))

        after = re.search(
            rf"판매가"
            rf"[\s:：|・\-]*"
            rf"{money}\s*원?",
            line,
        )

        if after:
            return int(after.group(1).replace(",", ""))

    return None

    # =====================================================
    # 2. 일반 판매가
    # =====================================================

    for line in lines:
        if "판매가" not in line:
            continue

        # 할인판매가는 위에서 이미 처리
        if "할인판매가" in line or "할인 판매가" in line:
            continue

        if "소비자가" in line:
            continue

        # -------------------------------------------------
        # 가격이 판매가보다 앞에 있는 형태
        #
        # 10,690원 판매가
        # 10,690원판매가
        # -------------------------------------------------
        before = re.search(
            rf"{money}\s*원?\s*판매가",
            line,
        )

        if before:
            price = to_int(before.group(1))
            print("  [가격근거/판매가]", line.strip())
            return price

        # -------------------------------------------------
        # 판매가 뒤에 가격이 있는 형태
        #
        # 판매가 10,690원
        # 판매가 | 10,690
        # 판매가 : 10,690원
        # -------------------------------------------------
        after = re.search(
            rf"판매가"
            rf"[\s:：|・\-]*"
            rf"(?:가격[\s:：|・\-]*)?"
            rf"{money}\s*원?",
            line,
        )

        if after:
            price = to_int(after.group(1))
            print("  [가격근거/판매가]", line.strip())
            return price

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
    if not text:
        return []

    matches = re.findall(
        r"(?<![A-Z0-9])([A-Z]{5,}\d+[A-Z0-9]*)(?![A-Z0-9])",
        text.upper(),
    )

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


def is_detail_url(url: str) -> bool:
    if not url:
        return False

    parsed = urlparse(url)

    path = parsed.path.strip("/")

    print(f"주소짧은지=  {path}")

    # 루트 도메인만 있는 경우 제외
    if not path:
        return False

    # 너무 짧은 일반 카테고리/메인성 URL 제외 가능
    return path.lower() not in {"main", "home", "index"}


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

        print("\n" + "=" * 80)
        print(f"[{idx}] {title}")
        print("내용:", text)
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

    # 1. 상세 페이지 URL만
    detail_candidates = [c for c in candidates if is_detail_url(c.get("url"))]

    if detail_candidates:
        lowest = min(
            detail_candidates,
            key=lambda x: x["price"],
        )

        highest = max(
            detail_candidates,
            key=lambda x: x["price"],
        )

        print("최저 가격 후보")
        print("-" * 80)
        print("가격 :", lowest["price"])
        print("제목 :", lowest["title"])
        print("URL  :", lowest["url"])
        print("재고 :", lowest["stock_status"])
        print("SKU  :", lowest["product_codes"])

        print("\n최대 가격 후보")
        print("-" * 80)
        print("가격 :", highest["price"])
        print("제목 :", highest["title"])
        print("URL  :", highest["url"])
        print("재고 :", highest["stock_status"])
        print("SKU  :", highest["product_codes"])

    else:
        print("가격 후보를 찾지 못했습니다.")

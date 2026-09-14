import os
import sqlite3
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# .env 파일의 SQL 값에는 SQLite 데이터베이스 파일 경로가 들어 있습니다.
sql_path = os.getenv("SQL")


def qdrant_to_sql(
    results: list[dict[str, Any]], table: str = "products"
) -> list[dict[str, Any]]:
    """Qdrant 검색 결과의 상품 ID로 SQLite 상품 정보를 조회합니다.

    Args:
        results: Qdrant가 반환한 검색 결과 목록입니다. 각 결과에는
            product_id가 필요하고, image_url은 없어도 됩니다.
        table: 상품을 조회할 SQLite 테이블 이름입니다.

    Returns:
        Qdrant 검색 순서대로 정렬된 상품 정보 목록입니다.
    """
    # 검색 결과가 없으면 SQL의 IN 절을 만들 수 없으므로 바로 빈 목록을 반환합니다.
    if not results:
        return []

    # 테이블 이름에는 SQL 파라미터(?)를 사용할 수 없으므로 안전한 이름인지 확인합니다.
    if not table.isidentifier():
        raise ValueError(f"Invalid table name: {table}")

    # DB 경로가 설정되지 않았다면 원인을 알 수 있도록 명확한 오류를 발생시킵니다.
    if not sql_path:
        raise RuntimeError("SQL environment variable is not set")

    # Qdrant 결과에서 조회할 상품 ID만 꺼냅니다.
    ids = [r["product_id"] for r in results]

    # ID 값을 SQL 문자열에 직접 넣지 않고 ? 자리에 안전하게 전달합니다.
    placeholders = ", ".join("?" for _ in ids)

    # CASE 문을 만들어 Qdrant의 검색 결과 순서가 DB 조회 후에도 유지되게 합니다.
    order_clause = " ".join(f"WHEN ? THEN {i}" for i, _ in enumerate(ids))

    # 상품 응답에 필요한 테이블 필드만 조회합니다.
    sql = f"""
        SELECT
            id,
            title,
            brand,
            category1,
            sale_price,
            discount_rate,
            image_url,
            tags,
            site,
            product_url,
            collected_at
        FROM "{table}"
        WHERE id IN ({placeholders})
        ORDER BY CASE id {order_clause} END
    """

    # 앞쪽 ids는 WHERE 절에, 뒤쪽 ids는 정렬용 CASE 문에 전달됩니다.
    with sqlite3.connect(sql_path) as conn:
        cursor = conn.execute(sql, [*ids, *ids])
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

    # DB에 이미지가 없을 때 사용할 Qdrant 이미지를 상품 ID별로 저장합니다.
    qdrant_images = {str(r["product_id"]): r.get("image_url", "") for r in results}
    rows_dict = []
    for row in rows:
        # SQLite의 튜플 형태 결과를 {필드명: 값} 형태의 딕셔너리로 바꿉니다.
        d = dict(zip(columns, row))

        # DB의 image_url이 비어 있으면 Qdrant 검색 결과의 이미지를 대신 사용합니다.
        if not d.get("image_url"):
            d["image_url"] = qdrant_images.get(str(d["id"]), "")
        rows_dict.append(d)

    return rows_dict

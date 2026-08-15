import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
import sqlite3


sql_path = os.getenv("SQL")

def qdrant_to_sql(results: list, table: str = "products") -> str:
    ids = [r["product_id"] for r in results]
    in_clause = ", ".join(f"'{pid}'" for pid in ids)
    order_clause = " ".join(f"WHEN '{pid}' THEN {i}" for i, pid in enumerate(ids))

    sql = f"""SELECT id, title, link, image_url, lprice, hprice, mall_name, brand, category1, category2 category3, category4 
            FROM {table}
            WHERE id IN ({in_clause})
            ORDER BY CASE id {order_clause} END;"""
            
    with sqlite3.connect(sql_path) as conn:
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

    # Qdrant의 image_url을 product_id 기준으로 매핑
    qdrant_images = {r["product_id"]: r["image_url"] for r in results}
    rows_dict = []
    for row in rows:
        d = dict(zip(columns, row))
        # SQLite에 image_url이 없으면 Qdrant 결과에서 가져오기
        if not d.get("image_url"):
            d["image_url"] = qdrant_images.get(str(d["id"]), "")
        rows_dict.append(d)
        
    return rows_dict
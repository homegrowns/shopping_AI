import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool

sql_db = os.getenv("SQL")

# SQLite 연결
db = SQLDatabase.from_uri(sql_db)

@tool
def execute_sql_query(query: str) -> str:
    """
    SQLite DB에 SQL 쿼리를 실행합니다.
    반드시 SELECT 문만 사용하세요.
    테이블 목록: products
    """
    try:
        result = db.run(query)
        return str(result)
    except Exception as e:
        return f"에러: {str(e)}"

@tool
def get_table_schema(table_name: str) -> str:
    """
    테이블 스키마(컬럼 정보)를 조회합니다.
    사용 가능한 테이블: products
    """
    return db.get_table_info([table_name])
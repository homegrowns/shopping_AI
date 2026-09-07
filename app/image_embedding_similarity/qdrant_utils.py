"""
Qdrant 접속/컬렉션/검색 유틸.
FastAPI, Airflow 양쪽에서 동일하게 사용합니다.
EC2 Qdrant 주소는 환경변수로 주입 (QDRANT_HOST, QDRANT_PORT).
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .embedder import VECTOR_SIZE

# .env 파일을 환경변수로 로드. 반드시 다른 모듈(qdrant_utils, s3_utils 등)을
# import하기 "전에" 실행되어야 한다 - 그 모듈들은 import되는 시점에
# os.getenv()로 QDRANT_HOST 등을 읽기 때문.
load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST", "EC2_PUBLIC_IP_HERE")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

COLLECTION_NAME_IMG = os.getenv("QDRANT_COLLECTION_IMG", "products_images")
# COLLECTION_NAME_HYBRID = os.getenv("QDRANT_COLLECTION_DESC", "products_hybrid")


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def upsert_points(client: QdrantClient, points: list[PointStruct]) -> None:
    if points:
        client.upsert(collection_name=COLLECTION_NAME_IMG, points=points)


def search_similar(client: QdrantClient, vector: list[float], top_k: int = 5):
    response = client.query_points(
        collection_name=COLLECTION_NAME_IMG,
        query=vector,
        limit=top_k,
    )
    return response.points

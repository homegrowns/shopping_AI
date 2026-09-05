import os

from dotenv import load_dotenv

load_dotenv()
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import aioboto3
import numpy as np
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from app.image_embedding_similarity.crop_utils import (
    crop_with_padding,
    get_crop_hint_box,
)
from app.image_embedding_similarity.embedder import ClipEmbedder
from app.image_embedding_similarity.label_utils import get_image_labels
from app.image_embedding_similarity.qdrant_utils import (
    ensure_collection,
    get_client,
    search_similar,
)
from app.rag.agent_stream import start_agent
from app.result_item_class import SearchResponse, SearchResultItem
from app.service.search_service import build_query_vector

BASE_DIR = Path(__file__).resolve().parent

print(BASE_DIR)

s3_session = aioboto3.Session()
S3_BUCKET = os.getenv("S3_BUCKET")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.qdrant_client = get_client()
    ensure_collection(app.state.qdrant_client)
    app.state.embedder = ClipEmbedder.get_instance()

    yield

    app.state.qdrant_client.close()


app = FastAPI(lifespan=lifespan)

FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "https://my-smart-shop.vercel.app",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"service": "Shopping AI Assistant API", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


async def _prepare_search(request, message, s3_key):
    """검색 전처리 로직 (이미지 바이트 확보, 라벨 감지, 쿼리 벡터 생성)"""
    contents = None
    message = message.strip() if message else None

    embedder = request.app.state.embedder

    # ── I/O: 이미지 바이트 확보 ──
    if s3_key:
        async with s3_session.client("s3") as s3:
            obj = await s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            contents = await obj["Body"].read()

    if not contents and not message:
        return None  # 에러 케이스

    if not message:
        print("메세지가 없습니다.")

    image_labels: list[str] = []
    if contents is not None:
        try:
            image_labels = get_image_labels(contents)
        except Exception as e:
            if "credentials" in str(e).lower() or "valid type" in str(e).lower():
                print(
                    f"[Label Detection] Google 인증 파일 문제로 실패, 라벨 없이 진행: {e}"
                )
            else:
                print(f"[Label Detection] 실패, 라벨 없이 진행: {e}")

    # ── 비즈니스 로직 ──
    query_vector, crop_applied, is_image_collection = build_query_vector(
        contents, message, embedder
    )

    if message is not None:
        agent_kwargs = dict(
            query_text=message,
            label_text=", ".join(image_labels[:5]),
            query_vector=query_vector,
            is_image_collection=is_image_collection,
        )
    else:
        agent_kwargs = dict(
            label_text="[이미지 속 카테고리]=" + ", ".join(image_labels[:5]),
            query_vector=query_vector,
            is_image_collection=is_image_collection,
        )

    return agent_kwargs


# ── 기존 프론트엔드 호환: JSON 응답 ──
@app.post("/search", response_model=SearchResponse)
async def search(
    request: Request,
    session_id: str = Form(...),
    message: str | None = Form(None),
    s3_key: str | None = Form(None),
):
    agent_kwargs = await _prepare_search(request, message, s3_key)
    if agent_kwargs is None:
        return JSONResponse(
            status_code=400,
            content={"error": "이미지 또는 텍스트 중 하나는 입력해야 합니다."},
        )

    # generator를 소비해서 최종 결과만 수집 (LLM 1회만 호출)
    search_results = []
    answer = ""
    for event in start_agent(**agent_kwargs):
        if event["type"] == "done":
            search_results = event["search_results"]
            answer = event["answer"]

    print(f"\n search_results: {search_results}")
    print(f"\n answer: {answer}")

    return SearchResponse(
        results=search_results,
        answer=answer,
    )


# ── SSE 스트리밍: 토큰 단위 실시간 전송 ──
@app.post("/search/stream")
async def search_stream(
    request: Request,
    session_id: str = Form(...),
    message: str | None = Form(None),
    s3_key: str | None = Form(None),
):
    agent_kwargs = await _prepare_search(request, message, s3_key)
    if agent_kwargs is None:
        return JSONResponse(
            status_code=400,
            content={"error": "이미지 또는 텍스트 중 하나는 입력해야 합니다."},
        )

    async def event_generator():
        async for event in start_agent(**agent_kwargs):
            if event["type"] == "status":
                # 프론트엔드가 data.status로 읽음 → onStatus 콜백 호출
                yield f"data: {json.dumps({'status': event['status']}, ensure_ascii=False)}\n\n"
            elif event["type"] == "token":
                # 프론트엔드가 data.token으로 읽음 → onToken 콜백 호출
                yield f"data: {json.dumps({'token': event['content']}, ensure_ascii=False)}\n\n"
            elif event["type"] == "done":
                # 프론트엔드가 data.done / data.results로 읽음 → onResults 콜백 호출
                yield f"data: {json.dumps({'done': True, 'results': event['search_results'], 'answer': event['answer']}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
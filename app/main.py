import os
from dotenv import load_dotenv

# .env 파일을 환경변수로 로드. 반드시 다른 모듈(qdrant_utils, s3_utils 등)을
# import하기 "전에" 실행되어야 한다 - 그 모듈들은 import되는 시점에
# os.getenv()로 QDRANT_HOST 등을 읽기 때문.
# 애플리케이션의 시작점(main.py)에서 딱 한 번만 .env를 로드하므로, 
# 하위 모듈들에는 load_dotenv() 코드를 쓰지 않아도 돼서 코드가 깔끔해집니다.
load_dotenv()

import io
import numpy as np
from fastapi import FastAPI, File, Form, Query, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from PIL import Image

from app.rag.agent import start_agent
from app.image_embedding_similarity.embedder import ClipEmbedder
from app.image_embedding_similarity.crop_utils import crop_with_padding, get_crop_hint_box
from app.image_embedding_similarity.qdrant_utils import ensure_collection, get_client, search_similar
from app.service.search_service import build_query_vector

import aioboto3
from contextlib import asynccontextmanager

BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)

s3_session = aioboto3.Session()
S3_BUCKET = os.getenv("S3_BUCKET")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 1회: Qdrant 클라이언트 + CLIP 모델 로드

    # ── startup ──
    app.state.qdrant_client = get_client()
    ensure_collection(app.state.qdrant_client)
    app.state.embedder = ClipEmbedder.get_instance()
    
    yield  # 서버 실행 중
    
    # ── shutdown ──
    app.state.qdrant_client.close()  # 리소스 정리

app = FastAPI(lifespan=lifespan)

templates = Jinja2Templates(directory=BASE_DIR / "templates")

@app.get("/", response_class=HTMLResponse)
async def search(request: Request, s: str | None = None):
    # print(q)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search")
async def search(
    request: Request,
    session_id: str = Form(...),
    message: str | None = Form(None),
    s3_key: str | None = Form(None),
):
    contents = None
    message = message.strip() if message else None

    embedder = request.app.state.embedder 

    # ── I/O: 이미지 바이트 확보 ──
    if s3_key:
        async with s3_session.client("s3") as s3:
            obj = await s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            contents = await obj["Body"].read()

        
    if not contents and not message:
        return JSONResponse(status_code=400, content={"error": "이미지 또는 텍스트 중 하나는 입력해야 합니다."})

    # ── 비즈니스 로직 ──
    query_vector, crop_applied, eng_text = build_query_vector(contents, message, embedder)
    # RAG 파이프시작
    final_state = start_agent(message, query_vector, eng_text)

    structured_results = final_state.get("search_results")

    print(f"\n final_state: {structured_results}")
    return JSONResponse(content={
        "results": structured_results
    })
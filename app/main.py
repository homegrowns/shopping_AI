import os
from dotenv import load_dotenv
load_dotenv()
import io
import numpy as np
import json
from fastapi import FastAPI, File, Form, Query, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware         
from pathlib import Path
from PIL import Image
import aioboto3
from contextlib import asynccontextmanager
from app.rag.agent import start_agent
from app.image_embedding_similarity.embedder import ClipEmbedder
from app.image_embedding_similarity.crop_utils import crop_with_padding, get_crop_hint_box
from app.image_embedding_similarity.qdrant_utils import ensure_collection, get_client, search_similar
from app.service.search_service import build_query_vector
from app.image_embedding_similarity.label_utils import get_image_labels

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

# CORS 미들웨어 추가
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "https://shopping-assistant-agent-front.vercel.app",  # Vercel 프론트엔드
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

from fastapi.templating import Jinja2Templates
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
    elif not message:
        print("메세지가 없습니다.")

    image_labels: list[str] = []
    if contents is not None:
        try:
            image_labels = get_image_labels(contents)
        except Exception as e:
            if "credentials" in str(e).lower() or "valid type" in str(e).lower():
                print(f"[Label Detection] Google 인증 파일 문제로 실패, 라벨 없이 진행: {e}")
            else:
                print(f"[Label Detection] 실패, 라벨 없이 진행: {e}")

    # ── 비즈니스 로직 ──
    query_vector, crop_applied, is_image_collection = build_query_vector(contents, message, embedder)

    # RAG 파이프시작
    if message is not None:
        final_state = start_agent(query_text=message, query_vector=query_vector, is_image_collection=is_image_collection)
    else:
        final_state = start_agent(label_text="[이미지 속 카테고리]="+", ".join(image_labels[:5]), query_vector=query_vector, is_image_collection=is_image_collection)

    structured_results = final_state.get("search_results")
    answer_text = final_state.get("answer", "")

    print(f"\n final_state: {structured_results}")
    print(f"\n answer_text: {answer_text}")

    ##!! stg 환경(Gemini)은 내부적으로 answer 값을 LangChain의 AIMessage 객체로 감싸서 돌려줄 수 있습니다. 
    # 이 객체는 겉보기에는 문자열처럼 print()도 잘 되고 로그에도 예쁘게 찍히지만, json.dumps()가 "이건 문자열이 아니라 
    # 복잡한 파이썬 객체잖아! 나는 이걸 JSON으로 못 바꿔!" 하며 500 에러를 뱉는 원인입니다.

    #dev 환경(ChatOllama)으로 테스트하셨다면, Ollama는 answer 값을 **순수한 파이썬 문자열(str)**로 깔끔하게 돌려줍니다. 
    # json.dumps()가 아무런 문제 없이 통과

    # ── 안전한 JSON 변환 처리 (500 에러 방지) ──
    # answer_text가 LangChain AIMessage 객체일 수 있으므로 문자열로 변환
    if hasattr(answer_text, 'content'):
        answer_text = answer_text.content
    answer_text = str(answer_text) if answer_text else ""

    # structured_results 안의 값들을 JSON 직렬화 가능한 타입으로 변환
    safe_results = []
    if structured_results:
        for item in structured_results:
            safe_item = {}
            for k, v in item.items():
                if hasattr(v, 'item'):      # numpy int64, float64 등
                    safe_item[k] = v.item()
                elif hasattr(v, 'content'):  # LangChain 객체
                    safe_item[k] = v.content
                else:
                    safe_item[k] = v
            safe_results.append(safe_item)

    return JSONResponse(content={
        "results": safe_results,
        "answer": answer_text
    })
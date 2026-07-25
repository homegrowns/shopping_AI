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

from contextlib import asynccontextmanager

BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)

# 크롭 이미지와 원본 이미지를 함께 임베딩할 때의 가중치
# (크롭만 쓰면 장식 디테일에 과도하게 집중되는 문제가 있어 원본 맥락을 일부 반영)
CROP_WEIGHT = 0.6
ORIGINAL_WEIGHT = 0.4

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


@app.get("/query", response_class=HTMLResponse)
def query(request: Request, s: str = "대한민국 수도는?"):
    print(s)
    result = start_agent(s)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"answer": result.get("messages")[-1].content},
    )

@app.post("/search")
async def search(
    request: Request,
    session_id: str = Form(...),
    message: str | None = Form(None),
    file: UploadFile | None = File(None),
    top_k: int = Query(default=5, ge=1, le=50),
):

    client = request.app.state.qdrant_client
    embedder = request.app.state.embedder   
    message = message.strip() if message else None

    if not file and not message:
        return JSONResponse(
            status_code=400,
            content={"error": "이미지 또는 텍스트 중 하나는 입력해야 합니다."},
        )

    image_vector = None
    text_vector = None
    s3_key = None
    crop_applied = False

    if file is not None:
        contents = await file.read()
        try:
            original_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception:
            return JSONResponse(
                status_code=400, content={"error": "이미지 파일을 읽을 수 없습니다."}
            )

        try:
            box = get_crop_hint_box(contents, original_image.size)
        except Exception as e:
            print(f"Crop Hints 실패, 원본 이미지로 진행: {e}")
            box = None

        if box:
            crop_image = crop_with_padding(original_image, box)
            crop_vector = np.array(embedder.embed_image(crop_image))
            original_vector = np.array(embedder.embed_image(original_image))
            combined_image_vector = (
                CROP_WEIGHT * crop_vector + ORIGINAL_WEIGHT * original_vector
            )
            combined_image_vector = combined_image_vector / np.linalg.norm(
                combined_image_vector
            )
            image_vector = combined_image_vector.tolist()
            crop_applied = True
        else:
            image_vector = embedder.embed_image(original_image)

    if message:
        text_vector = embedder.embed_text(message)

    # 3) 이미지+텍스트 둘 다 있으면 50:50 평균(재정규화), 하나만 있으면 그대로 사용
    if image_vector is not None and text_vector is not None:
        combined = (np.array(image_vector) + np.array(text_vector)) / 2
        combined = combined / np.linalg.norm(combined)
        query_vector = combined.tolist()
    else:
        query_vector = image_vector if image_vector is not None else text_vector

    results = search_similar(client, query_vector, top_k=top_k)

    payload = [
        {
            "score": round(r.score, 4),
            "product_id": r.payload.get("product_id"),
            "title": r.payload.get("title"),
            "image_url": r.payload.get("image_url"),
        }
        for r in results
    ]
    return JSONResponse(
        content={
            "results": payload,
            "s3_key": s3_key,
            "crop_applied": crop_applied,
        }
    )
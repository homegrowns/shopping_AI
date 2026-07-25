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
# [변경] feature/label-text: 이미지 검색 결과를 자연어로 정리/필터링하는 RAG 함수 2개 추가
from app.rag.nodes import filter_products_by_labels, generate_shopping_answer
from app.image_embedding_similarity.embedder import ClipEmbedder
from app.image_embedding_similarity.crop_utils import crop_with_padding, get_crop_hint_box
# [변경] feature/label-text: Vision Label Detection 유틸 추가
from app.image_embedding_similarity.label_utils import get_image_labels
from app.image_embedding_similarity.qdrant_utils import ensure_collection, get_client, search_similar

from contextlib import asynccontextmanager

BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)

# 크롭 이미지와 원본 이미지를 함께 임베딩할 때의 가중치
# (크롭만 쓰면 장식 디테일에 과도하게 집중되는 문제가 있어 원본 맥락을 일부 반영)
CROP_WEIGHT = 0.6
ORIGINAL_WEIGHT = 0.4

# [변경] feature/label-text: 이 점수 미만인 검색 결과는 아예 제외.
# 카테고리 필터링은 여기서 하지 않고 유사도 점수로만 1차 거른 뒤,
# 실제 카테고리 판단은 generate_shopping_answer()의 LLM 프롬프트에서 처리한다.
MIN_SCORE = 0.8

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
    # [변경] feature/label-text: 기본값 5 -> 20 (score_threshold가 실제 필터
    # 역할을 하므로, top_k는 "최대 몇 개까지"로 의미가 바뀜)
    top_k: int = Query(default=20, ge=1, le=50, description="반환할 최대 개수(상한)"),
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

    # [신규] feature/label-text: 이미지가 있으면 Label Detection으로 라벨을
    # 뽑아둔다. 검색 쿼리 벡터에는 반영하지 않는다(라벨은 대체로 광범위한
    # 카테고리 단어라 image_vector와 섞으면 검색 방향이 흐려질 위험이 있어서).
    # 대신 generate_shopping_answer()의 프롬프트에 참고 컨텍스트로만 전달한다.
    image_labels: list[str] = []
    if file is not None:
        try:
            image_labels = get_image_labels(contents)
        except Exception as e:
            print(f"[Label Detection] 실패, 라벨 없이 진행: {e}")

    # 3) 이미지+텍스트 둘 다 있으면 50:50 평균(재정규화), 하나만 있으면 그대로 사용
    if image_vector is not None and text_vector is not None:
        combined = (np.array(image_vector) + np.array(text_vector)) / 2
        combined = combined / np.linalg.norm(combined)
        query_vector = combined.tolist()
    else:
        query_vector = image_vector if image_vector is not None else text_vector

    # [변경] feature/label-text: score_threshold=MIN_SCORE 추가 (baseline은 필터 없이 top_k만 사용)
    results = search_similar(client, query_vector, top_k=top_k, score_threshold=MIN_SCORE)

    payload = [
        {
            "score": round(r.score, 4),
            "product_id": r.payload.get("product_id"),
            "title": r.payload.get("title"),
            "image_url": r.payload.get("image_url"),
        }
        for r in results
    ]

    # [신규] feature/label-text: 텍스트 질문 유무에 따라 처리 방식이 갈린다.
    # - 텍스트가 있으면: 카드는 스코어로만 거른 전체를 그대로 반환하고,
    #   카테고리 판단은 자연어 답변(generate_shopping_answer)에서만 다룬다.
    # - 텍스트 없이 이미지만 있으면: 카드 목록 자체를 라벨과 의미적으로
    #   연관된 것만 추려서 반환한다 (filter_products_by_labels).
    # baseline에는 이 answer 생성 로직 자체가 없었음 (results만 그대로 반환).
    if message:
        answer = generate_shopping_answer(message, payload, image_labels=image_labels)
    else:
        relevant_ids = filter_products_by_labels(image_labels, payload)
        if relevant_ids is None:
            # 라벨이 없거나 LLM 호출 자체가 실패해 "판단 불가" -> 원본 후보 그대로 사용
            print("[Filter By Labels] 판단 불가, 원본 후보를 그대로 사용합니다.")
        else:
            # relevant_ids가 빈 리스트여도 이는 "정말 연관 상품이 없다"는
            # LLM의 판단 결과이므로 그대로 반영한다 (원본으로 되돌리지 않음)
            payload = [p for p in payload if p["product_id"] in relevant_ids]

        effective_question = "첨부한 이미지와 유사하면서 관련 있는 상품을 추천해줘"
        answer = generate_shopping_answer(
            effective_question, payload, image_labels=image_labels
        )

    return JSONResponse(
        content={
            "answer": answer,  # [변경] feature/label-text: 신규 필드
            "results": payload,
            "s3_key": s3_key,
            "crop_applied": crop_applied,
            "image_labels": image_labels,  # [변경] feature/label-text: 신규 필드
        }
    )
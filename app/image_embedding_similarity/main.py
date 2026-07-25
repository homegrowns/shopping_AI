"""
로컬에서 실행하는 FastAPI 서버.
- EC2 위 Qdrant에 원격 접속 (QDRANT_HOST 환경변수)
- 사용자가 이미지/텍스트(또는 둘 다)를 입력 -> 유사 상품 검색 -> 결과 반환

처리 흐름:
1) 이미지가 있으면: 원본을 S3(input/{session_id}/{n}.jpg)에 저장
2) 이미지가 있으면: Google Vision Crop Hints로 시각적으로 중요한 영역 감지
   - 감지되면: 크롭 이미지와 원본 이미지를 각각 임베딩한 뒤
     크롭 60% : 원본 40% 가중 평균(재정규화)해서 사용
     (크롭만 쓰면 장식 디테일에 과도하게 집중해 상품 전체 맥락을 잃는
     경우가 있어, 원본 맥락을 40% 비중으로 함께 반영한다)
   - 감지 안 되면(또는 조건 미달): 원본 이미지만 임베딩
3) 텍스트가 있으면: CLIP 텍스트 인코더로 임베딩
4) 이미지+텍스트 둘 다 있으면: (2번의 이미지 벡터) + 텍스트 벡터를
   50:50 평균(재정규화)해서 검색 쿼리로 사용. 하나만 있으면 그대로 사용

실행: uv run uvicorn main:app --reload --port 8000 --app-dir .

필요 환경변수:
- QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION
- GOOGLE_APPLICATION_CREDENTIALS (Google Vision 서비스 계정 키 경로)
- AWS 자격증명 (boto3 기본 방식: 환경변수 또는 ~/.aws/credentials)
"""
import io

from dotenv import load_dotenv

# .env 파일을 환경변수로 로드. 반드시 다른 모듈(qdrant_utils, s3_utils 등)을
# import하기 "전에" 실행되어야 한다 - 그 모듈들은 import되는 시점에
# os.getenv()로 QDRANT_HOST 등을 읽기 때문.
load_dotenv()

import numpy as np
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from embedder import ClipEmbedder
from crop_utils import crop_with_padding, get_crop_hint_box
from qdrant_utils import ensure_collection, get_client, search_similar
from s3_utils import upload_input_image

app = FastAPI(title="이미지+텍스트 유사도 검색 API")

# 크롭 이미지와 원본 이미지를 함께 임베딩할 때의 가중치
# (크롭만 쓰면 장식 디테일에 과도하게 집중되는 문제가 있어 원본 맥락을 일부 반영)
CROP_WEIGHT = 0.6
ORIGINAL_WEIGHT = 0.4

# 서버 시작 시 1회: Qdrant 클라이언트 + CLIP 모델 로드
qdrant_client = get_client()
ensure_collection(qdrant_client)
embedder = ClipEmbedder.get_instance()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
async def search(
    session_id: str = Form(...),
    message: str | None = Form(None),
    file: UploadFile | None = File(None),
    top_k: int = Query(default=5, ge=1, le=50),
):
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

        # 1) 원본 이미지를 S3에 저장 (세션별 순번 파일명, 사용자가 입력한 그대로 저장)
        try:
            s3_key = upload_input_image(session_id, original_image)
        except Exception as e:
            print(f"S3 업로드 실패 (검색은 계속 진행): {e}")

        # 2) Crop Hints로 시각적으로 중요한 영역 감지
        #    -> 감지되면 크롭+원본을 6:4로 가중 평균, 감지 안 되면 원본만 사용
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

    results = search_similar(qdrant_client, query_vector, top_k=top_k)

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
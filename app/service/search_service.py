import io
import numpy as np
from PIL import Image
from app.image_embedding_similarity.embedder import ClipEmbedder
from app.image_embedding_similarity.crop_utils import get_crop_hint_box, crop_with_padding

from deep_translator import GoogleTranslator

# 크롭 이미지와 원본 이미지를 함께 임베딩할 때의 가중치
# (크롭만 쓰면 장식 디테일에 과도하게 집중되는 문제가 있어 원본 맥락을 일부 반영)
CROP_WEIGHT = 0.6
ORIGINAL_WEIGHT = 0.4

def build_query_vector(
    contents: bytes | None,
    message: str | None,
    embedder: ClipEmbedder,
) -> tuple[list[float], bool, bool]:
    """
    이미지 바이트 + 텍스트 → 최종 검색 벡터 생성.
    Returns: (query_vector, crop_applied, eng_text)
    """
    image_vector = None
    text_vector = None
    original_image = None
    box = None
    msg = ''
    crop_applied = False
    eng_text = False

    # 100% 영어(알파벳/숫자/기호)인지 확인
    if message is not None and message.isascii():
        print(f"[번역 생략] 영문 감지됨, 번역 없이 진행: {message}")
        msg = message
        eng_text = True
    #영어가 아닌경우 번역
    elif message is not None and message.isascii()==False:
        msg = GoogleTranslator(source='ko', target='en').translate(message)
        print(f"[google 번역] {message} -> {msg} \n") # 디버그용 출력 
        eng_text = True
    
    # 메시지에서 " that" 앞까지만 잘라와서 text_vector로 사용
    index = msg.find(" that")
    if index != -1:
        result = msg[:index]
        print(result)  # 출력: Recommended jeans
        text_vector = embedder.embed_text(result)
    else:
        print("that 키워드가 없습니다. 전체 쿼리를 벡터화 합니다.") 
        text_vector = embedder.embed_text(msg)

    if contents is not None:
        original_image = Image.open(io.BytesIO(contents)).convert("RGB")

        try:
            box = get_crop_hint_box(contents, original_image.size)
        except Exception:
            box = None

    if box and contents is not None:
        # 원본 이미지에서 박스 영역을 패딩 포함하여 크롭
        crop_image = crop_with_padding(original_image, box)

        # 크롭된 이미지의 임베딩 벡터 생성
        crop_vec = np.array(embedder.embed_image(crop_image))

        # 원본 이미지 전체의 임베딩 벡터 생성
        orig_vec = np.array(embedder.embed_image(original_image))

        # 크롭 벡터와 원본 벡터를 가중합으로 결합
        # (CROP_WEIGHT: 크롭 이미지 가중치, ORIGINAL_WEIGHT: 원본 이미지 가중치)
        combined = CROP_WEIGHT * crop_vec + ORIGINAL_WEIGHT * orig_vec

        # L2 정규화 (단위 벡터로 변환하여 코사인 유사도 검색에 적합하게 만듦)
        combined = combined / np.linalg.norm(combined)
        image_vector = combined.tolist()
        crop_applied = True

    elif contents is not None:
        # 박스가 없으면 원본 이미지 전체를 그대로 임베딩
        image_vector = embedder.embed_image(original_image)

    # 이미지 벡터와 텍스트 벡터가 모두 존재하는 경우
    if image_vector and text_vector:
# TODO: visoin label text 를 나중에 주입하여 text_vector 유사도 검사에 도움주는 기능
#      1. 이미지와 텍스트질문이 찾는 카테고리가 같으면 이미지를 벡터화후 query_vector에 활용
        query_vector = text_vector
    if " than" in msg:
        print("두 벡터의 평균을 구해 멀티모달 쿼리 벡터 생성.") 
        # 두 벡터의 평균을 구해 멀티모달 쿼리 벡터 생성
        combined = (np.array(image_vector) + np.array(text_vector)) / 2
        # L2 정규화
        combined = combined / np.linalg.norm(combined)
        query_vector = combined.tolist()
    else:
        # 둘 중 하나만 존재하면 그것을 쿼리 벡터로 사용
        query_vector = image_vector or text_vector
    # 최종 쿼리 벡터와 크롭 적용 여부를 반환
    return query_vector, crop_applied, eng_text
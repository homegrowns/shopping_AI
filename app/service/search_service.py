import io
import re
import numpy as np
from typing import Optional, Tuple, List, Any
from PIL import Image
from app.image_embedding_similarity.embedder import ClipEmbedder
from app.image_embedding_similarity.crop_utils import get_crop_hint_box, crop_with_padding

from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# 크롭 이미지와 원본 이미지를 함께 임베딩할 때의 가중치
# (크롭만 쓰면 장식 디테일에 과도하게 집중되는 문제가 있어 원본 맥락을 일부 반영)
CROP_WEIGHT = 0.6
ORIGINAL_WEIGHT = 0.4

# 검사하고 싶은 단어들을 리스트로 모아둡니다.
KEYWORD = (
    "than",
    "same",
    "with",
    "match",
    "matching",
    "in",
    "similar",
    "color",
    "suitable",
    "look good on",
    "brighter",
    "darker",
    "fit",
)

CATEGORY_ALIASES = {
    # 원피스
    "one piece": "dress",
    "onepiece": "dress",
    "one-piece": "dress",

    # 티셔츠
    "tee": "t-shirt",
    "tea": "t-shirt",

    # 맨투맨
    "mtm": "sweatshirt",
    "sweat shirt": "sweatshirt",

    # 후드티
    "hooded sweatshirt": "hoodie",
    "hooded": "hoodie",

    # 청바지
    "denim": "jeans",
    "denim pants": "jeans",
    

    # 슬랙스
    "slacks": "pants",
    "trousers": "pants",

    # 운동화
    "trainers": "sneakers",
    "trainer": "sneakers",
    "running shoes": "sneakers",

    # 점퍼
    "jumper": "jacket",   # 한국에서 말하는 점퍼 기준

    # 패딩
    "puffer": "padding",
    "puffer jacket": "padding",
    "down jacket": "padding",

    # 니트
    "knitwear": "knit",
    "sweater": "knit",

    # 와이셔츠
    "dress shirt": "shirt",

    # 백팩
    "back pack": "backpack",
}

model = SentenceTransformer("clip-ViT-B-32", device="cuda")

def get_text_vector(msg: str, embedder):
    """
    메시지에서 " that" 앞부분만 추출하여 소문자로 바꾼 뒤 벡터화하는 함수.
    " that" 키워드가 없으면 전체 메시지를 소문자로 변환하여 벡터화합니다.
    """
    index = msg.find(" that") or msg.find(" matching")
    
    if any(category in msg for category in CATEGORY_ALIASES):
        for alias, category in CATEGORY_ALIASES.items():
            msg = msg.replace(alias, category)
            
    if index != -1:
        result = msg[:index].lower()
        print(f"that 키워드가 있습니다. 추출된 텍스트: {result}\n")
        return embedder.embed_text(result)
    else:
        print(f"that 키워드가 없습니다. 전체 질문을 벡터화 합니다. : {msg}\n")
        return embedder.embed_text(msg.lower())

def get_image_vector(
    original_image: Any,         # PIL 사용 시: Image.Image / OpenCV 사용 시: np.ndarray
    box: Optional[List[int]],    # [x1, y1, x2, y2] 형태의 리스트나 튜플 (없을 수도 있으니 Optional)
    contents: Optional[Any],     # 추출된 텍스트나 결과값 (타입에 맞춰 str 등으로 변경 가능)
    embedder: Any                # 사용 중이신 Embedder 클래스 객체
) -> Tuple[Optional[List[float]], bool]:
    """
    바운딩 박스 유무에 따라 이미지를 크롭하여 가중합 벡터를 만들거나, 
    박스가 없으면 원본 이미지 전체를 벡터화하여 반환하는 함수.
    
    Returns:
        Tuple[Optional[List[float]], bool]: (이미지 벡터 리스트, 크롭 적용 여부)
    """
    image_vector: Optional[List[float]] = None
    crop_applied: bool = False
    
    if box and contents is not None:
        crop_image = crop_with_padding(original_image, box)

        crop_vec = np.array(embedder.embed_image(crop_image))
        orig_vec = np.array(embedder.embed_image(original_image))

        combined = CROP_WEIGHT * crop_vec + ORIGINAL_WEIGHT * orig_vec

        combined = combined / np.linalg.norm(combined)
        
        image_vector = combined.tolist()
        crop_applied = True

    elif contents is not None:
        print("이미지에서 박스가 검출되지 않았습니다. 전체 이미지를 벡터화 합니다.\n")
        image_vector = embedder.embed_image(original_image)
        # crop_applied는 False 유지

    return image_vector, crop_applied


def build_query_vector(
    contents: bytes | None,
    message: str | None,
    embedder: ClipEmbedder,
) -> tuple[list[float], bool]:
    """
    이미지 바이트 + 텍스트 → 최종 검색 벡터 생성.
    Returns: (query_vector, crop_applied)
    """
    image_vector = None
    text_vector = None
    original_image = None
    box = None
    msg = ''
    crop_applied = False

    # 100% 영어(알파벳/숫자/기호)인지 확인
    if message is not None and message.isascii():
        print(f"[번역 생략] 영문 감지됨, 번역 없이 진행: {message}")
        msg = message
        msg =  msg.lower()
        text_vector = get_text_vector(msg, embedder)
    #영어가 아닌경우 번역
    elif message is not None and message.isascii()==False:
        msg = GoogleTranslator(source='ko', target='en').translate(message)
        msg =  msg.lower()
        print(f"[google 번역] {message} -> {msg} \n") # 디버그용 출력 
        text_vector = get_text_vector(msg, embedder)   

    if contents is not None:
        original_image = Image.open(io.BytesIO(contents)).convert("RGB")

        try:
            box = get_crop_hint_box(contents, original_image.size)
        except Exception:
            box = None
    else:
        print("이미지가 없습니다.")
    
    # TODO: 2026-08-03
    # crop_applied 대신 이미지백터 bool로 [QDRANT SEARCH]툴 100%쓰도록 한다
    image_vector, crop_applied = get_image_vector(original_image, box, contents, embedder)

    # 이미지 벡터와 텍스트 벡터가 모두 존재하는 경우
    if image_vector and text_vector:
        print("텍스트(90%)와 이미지(10%)의 가중합으로 멀티모달 쿼리 벡터 생성.\n\n") 
            
        # 이미지 벡터에 0.4, 텍스트 벡터에 0.6을 곱해서 더해줍니다.
        combined = (0.1 * np.array(image_vector)) + (0.9 * np.array(text_vector))
        
        # L2 정규화 (길이를 1로 맞춰서 코사인 유사도 검색에 최적화)
        combined = combined / np.linalg.norm(combined)
        query_vector = combined.tolist()

# # TODO: visoin label text 를 나중에 주입하여 text_vector 유사도 검사에 도움주는 기능
# #      1. 이미지와 텍스트질문이 찾는 카테고리가 같으면 이미지를 벡터화후 query_vector에 활용
# #      2. 찾는 카테고리가 다르면 이미지의 label text에서 카테고리와 text 조합후 벡터화
        # print("텍스트 벡터를 사용해서 검색합니다.\n") 
        # query_vector = text_vector

        # # 리스트 안의 단어 중 하나라도(any) msg 안에 들어있는지 확인합니다.
        # if any(k in msg for k in KEYWORD): #예} 이 바지 보다 밝은
        #     print("텍스트(90%)와 이미지(10%)의 가중합으로 멀티모달 쿼리 벡터 생성.\n\n") 
            
        #     # 이미지 벡터에 0.2, 텍스트 벡터에 0.8을 곱해서 더해줍니다.
        #     combined = (0.1 * np.array(image_vector)) + (0.9 * np.array(text_vector))
            
        #     # L2 정규화 (길이를 1로 맞춰서 코사인 유사도 검색에 최적화)
        #     combined = combined / np.linalg.norm(combined)
        #     query_vector = combined.tolist()
    else:
        # 둘 중 하나만 존재하면 그것을 쿼리 벡터로 사용
        query_vector = image_vector or text_vector
    
    # 최종 쿼리 벡터와 크롭 적용 여부를 반환
    return query_vector, crop_applied
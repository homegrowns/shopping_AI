"""
Google Cloud Vision API로 이미지 안의 "의미 있는" 텍스트 영역을 감지한다.

기존 text_detection() 대신 document_text_detection()을 사용하는 이유:
- text_detection()은 전체 텍스트를 감싸는 하나의 박스만 주고, 신뢰도(confidence) 정보가 없다.
- document_text_detection()은 단어(word) 단위로 confidence 점수를 제공해서,
  피부/배경 패턴을 텍스트로 잘못 인식한 저신뢰 오탐지를 걸러낼 수 있다.

"의미 있는 텍스트"로 인정하는 조건 (모두 만족해야 크롭 진행):
1) 신뢰도(confidence)가 MIN_CONFIDENCE 이상인 단어만 사용
2) 그렇게 걸러진 단어들의 총 글자 수가 MIN_TOTAL_CHARS 이상
3) 감지된 영역이 원본 이미지 면적의 MIN_AREA_RATIO ~ MAX_AREA_RATIO 사이
   (너무 작으면 노이즈, 너무 크면 크롭하는 의미가 없음)
위 조건을 하나라도 못 만족하면 None을 반환하고, 호출부는 원본 이미지를 그대로 사용한다.

사전 준비 필요:
1) GCP 프로젝트에서 Cloud Vision API 활성화 + 결제 계정 연결
2) 서비스 계정 키(JSON) 발급 후 GOOGLE_APPLICATION_CREDENTIALS 환경변수 지정
3) 패키지 설치: uv pip install google-cloud-vision
"""
from PIL import Image

MIN_CONFIDENCE = 0.75      # 이 신뢰도 미만인 단어는 무시
MIN_TOTAL_CHARS = 4        # 신뢰도 높은 단어들을 합쳐도 이보다 짧으면 오탐지로 간주
MIN_AREA_RATIO = 0.05      # 원본 이미지 면적의 5% 미만이면 노이즈로 간주
MAX_AREA_RATIO = 0.85      # 원본 이미지 면적의 85% 초과면 크롭 의미 없음


def get_meaningful_text_crop(
    image_bytes: bytes,
    image_size: tuple[int, int],
) -> tuple[tuple[int, int, int, int], str] | None:
    """
    이미지에서 신뢰도 높은 텍스트 영역을 감지해 (box, detected_text)를 반환한다.
    조건을 만족하는 텍스트가 없으면 None을 반환한다.
    """
    from dotenv import load_dotenv
    from google.cloud import vision
# .env 파일에서 GOOGLE_APPLICATION_CREDENTIALS 환경 변수 로드
    load_dotenv()

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"Vision API 오류: {response.error.message}")

    full_text_annotation = response.full_text_annotation
    if not full_text_annotation or not full_text_annotation.pages:
        print("[OCR] 감지된 텍스트 없음")
        return None

    # 신뢰도 높은 단어만 수집
    confident_words: list[tuple[str, list]] = []
    for page in full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    if word.confidence < MIN_CONFIDENCE:
                        continue
                    word_text = "".join(symbol.text for symbol in word.symbols)
                    confident_words.append((word_text, word.bounding_box.vertices))

    if not confident_words:
        print("[OCR] 신뢰도 기준을 만족하는 단어 없음 (오탐지로 판단, 원본 사용)")
        return None

    combined_text = " ".join(w[0] for w in confident_words)
    total_chars = len(combined_text.replace(" ", ""))
    print(f"[OCR] 신뢰도 통과 텍스트: {combined_text!r} (총 {total_chars}자)")

    if total_chars < MIN_TOTAL_CHARS:
        print(f"[OCR] 글자 수 부족({total_chars} < {MIN_TOTAL_CHARS}), 원본 사용")
        return None

    xs = [v.x for _, verts in confident_words for v in verts]
    ys = [v.y for _, verts in confident_words for v in verts]
    box = (min(xs), min(ys), max(xs), max(ys))

    w, h = image_size
    box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    image_area = w * h
    ratio = box_area / image_area if image_area else 0
    print(f"[OCR] 크롭 박스: {box}, 면적 비율: {ratio:.1%}")

    if ratio < MIN_AREA_RATIO:
        print(f"[OCR] 영역이 너무 작음({ratio:.1%} < {MIN_AREA_RATIO:.0%}), 원본 사용")
        return None
    if ratio > MAX_AREA_RATIO:
        print(f"[OCR] 영역이 너무 큼({ratio:.1%} > {MAX_AREA_RATIO:.0%}), 크롭 의미 없어 원본 사용")
        return None

    return box, combined_text


def crop_with_padding(
    image: Image.Image,
    box: tuple[int, int, int, int],
    padding_ratio: float = 0.08,
) -> Image.Image:
    """
    감지된 텍스트 박스 주변에 여유(padding)를 살짝 두고 크롭한다.
    텍스트를 너무 타이트하게 자르면 주변 시각적 맥락(로고, 배경 등)이
    사라져 임베딩 품질이 떨어질 수 있어 약간의 여백을 둔다.
    """
    w, h = image.size
    left, top, right, bottom = box

    pad_x = int((right - left) * padding_ratio)
    pad_y = int((bottom - top) * padding_ratio)

    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(w, right + pad_x)
    bottom = min(h, bottom + pad_y)

    if right <= left or bottom <= top:
        return image

    return image.crop((left, top, right, bottom))
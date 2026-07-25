"""
Google Cloud Vision의 Crop Hints 기능으로 이미지에서
시각적으로 중요한(salient) 영역을 찾아 크롭 박스를 계산한다.

Crop Hints는 텍스트 유무와 무관하게, 배경보다 도드라지는 핵심 피사체
영역을 추천해준다. 상품 사진에서 배경을 배제하고 상품 자체에 집중해서
임베딩하기 위한 용도로 사용한다.

"의미 있는 크롭"으로 인정하는 조건 (모두 만족해야 크롭 진행):
1) 추천된 크롭 영역의 confidence가 MIN_CONFIDENCE 이상
2) 영역이 원본 이미지 면적의 MIN_AREA_RATIO ~ MAX_AREA_RATIO 사이
   (너무 작으면 노이즈, 너무 크면(거의 전체) 크롭하는 의미가 없음)
조건을 만족하지 못하면 None을 반환하고, 호출부는 원본 이미지를 그대로 사용한다.

사전 준비 필요:
1) GCP 프로젝트에서 Cloud Vision API 활성화 + 결제 계정 연결
2) 서비스 계정 키(JSON) 발급 후 GOOGLE_APPLICATION_CREDENTIALS 환경변수 지정
3) 패키지 설치: uv pip install google-cloud-vision
"""
from PIL import Image

MIN_CONFIDENCE = 0.5     # 이 신뢰도 미만인 크롭 추천은 무시
MIN_AREA_RATIO = 0.05    # 원본 이미지 면적의 5% 미만이면 노이즈로 간주
MAX_AREA_RATIO = 0.95    # 원본 이미지 면적의 95% 초과면 크롭 의미 없음


def get_crop_hint_box(
    image_bytes: bytes,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """
    Crop Hints API로 시각적으로 중요한 영역의 (left, top, right, bottom)
    박스를 반환한다. 조건을 만족하는 영역이 없으면 None을 반환한다.
    """
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.crop_hints(image=image)

    if response.error.message:
        raise RuntimeError(f"Vision API 오류: {response.error.message}")

    hints = response.crop_hints_annotation.crop_hints
    if not hints:
        print("[CropHints] 추천 영역 없음, 원본 사용")
        return None

    best = max(hints, key=lambda h: h.confidence)
    print(f"[CropHints] 최고 신뢰도: {best.confidence:.2f}")

    if best.confidence < MIN_CONFIDENCE:
        print(f"[CropHints] 신뢰도 낮음({best.confidence:.2f} < {MIN_CONFIDENCE}), 원본 사용")
        return None

    vertices = best.bounding_poly.vertices
    xs = [v.x for v in vertices]
    ys = [v.y for v in vertices]
    box = (min(xs), min(ys), max(xs), max(ys))

    w, h = image_size
    box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    image_area = w * h
    ratio = box_area / image_area if image_area else 0
    print(f"[CropHints] 크롭 박스: {box}, 면적 비율: {ratio:.1%}")

    if ratio < MIN_AREA_RATIO:
        print(f"[CropHints] 영역이 너무 작음({ratio:.1%} < {MIN_AREA_RATIO:.0%}), 원본 사용")
        return None
    if ratio > MAX_AREA_RATIO:
        print(f"[CropHints] 영역이 거의 전체({ratio:.1%} > {MAX_AREA_RATIO:.0%}), 크롭 의미 없어 원본 사용")
        return None

    return box


def crop_with_padding(
    image: Image.Image,
    box: tuple[int, int, int, int],
    padding_ratio: float = 0.05,
) -> Image.Image:
    """
    추천된 크롭 박스 주변에 약간의 여유(padding)를 두고 크롭한다.
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

"""
Google Cloud Vision Label Detection으로 이미지 안의 대략적인 내용(라벨)을 추출한다.
이미지+텍스트를 함께 입력했을 때, 사용자가 입력한 텍스트에 이 라벨과
겹치는 키워드가 있으면 검색 결과 재정렬 시 가중치를 주기 위한 용도로 사용한다.

주의: Vision API의 라벨은 기본적으로 영어로 반환된다. 사용자가 한국어로만
질문하면 겹침이 감지되지 않을 수 있다 (필요하면 추후 번역 단계 추가 가능).
"""
MIN_LABEL_CONFIDENCE = 0.6
MAX_LABELS = 15


def get_image_labels(image_bytes: bytes) -> list[str]:
    """
    이미지에서 신뢰도 MIN_LABEL_CONFIDENCE 이상인 라벨들을 소문자로 반환한다.
    """
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.label_detection(image=image, max_results=MAX_LABELS)

    if response.error.message:
        raise RuntimeError(f"Vision API 오류: {response.error.message}")

    labels = [
        annotation.description.lower()
        for annotation in response.label_annotations
        if annotation.score >= MIN_LABEL_CONFIDENCE
    ]
    print(f"[Label Detection] 감지된 라벨: {labels}")
    return labels


def find_overlapping_keywords(image_labels: list[str], message: str) -> list[str]:
    """
    이미지 라벨과 사용자 입력 텍스트 사이에 겹치는 키워드를 찾는다.
    - 라벨 전체가 텍스트 안에 부분 문자열로 포함되어 있거나
    - 라벨의 단어 중 하나라도 텍스트의 단어와 겹치면 매칭으로 간주한다
      (예: 라벨 "high heels" vs 메시지에 포함된 "heels")
    """
    message_lower = message.lower()
    message_tokens = set(message_lower.split())

    overlaps = []
    for label in image_labels:
        label_tokens = set(label.split())
        if label in message_lower or (label_tokens & message_tokens):
            overlaps.append(label)

    if overlaps:
        print(f"[Label Detection] 텍스트와 겹치는 키워드: {overlaps}")
    else:
        print(
            f"[Label Detection] 겹치는 키워드 없음 "
            f"(이미지 라벨: {image_labels}, 사용자 텍스트: {message!r})"
        )
    return overlaps
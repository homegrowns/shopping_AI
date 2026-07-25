# === feature/label-text =========================================
# Google Vision Label Detection으로 이미지의 대략적인 내용(라벨)을 뽑아내기 위해 새로 추가
#
# 처음엔 "라벨과 사용자 텍스트가 문자열로 겹치는지"를 코드로 직접 비교해서
# 검색 결과를 재정렬하는 방식(find_overlapping_keywords)으로 만들었었는데,
# - 라벨이 영어라 한국어 질문과 문자열이 거의 안 겹치고
# - "이런 거 찾아줘"처럼 애초에 비교할 단어가 없는 질문도 많아서
# 실효성이 없어 제거함. 지금은 라벨을 추출만 해서 RAG 프롬프트에 참고
# 컨텍스트로 넘기고, "의미가 겹치는지"는 LLM이 판단하게 함
# ============================================================================
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

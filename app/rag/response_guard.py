"""사용자에게 전달하기 직전의 응답에서 불필요한 안내를 제거합니다.

이미지를 첨부한 사용자가 이미지를 바탕으로 질문했는데도 모델이 다시 이미지를
첨부하거나 질문을 재전송하라고 답하는 경우가 있습니다. 이 모듈은 그런 문장만
후처리 단계에서 제거하여, 나머지 유효한 답변은 그대로 사용자에게 전달합니다.
"""

import re


# 마침표·느낌표·물음표 뒤의 공백 또는 줄바꿈을 문장 경계로 간주합니다.
# lookbehind를 사용하므로 문장부호 자체는 분리 과정에서 사라지지 않습니다.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")

# 모델이 이미지 제출을 요구할 때 자주 사용하는 표현들입니다.
_UPLOAD_TERMS = ("첨부", "업로드", "올려")

# 단순히 이미지를 언급한 설명까지 삭제하지 않도록, 재시도 요청을 나타내는
# 표현이 함께 존재하는지를 추가로 확인합니다.
_RETRY_TERMS = ("질문", "보내", "검색", "찾")


def suppress_redundant_image_request(answer: str, has_image: bool) -> str:
    """이미지가 이미 첨부된 경우 재첨부를 요구하는 문장만 제거합니다.

    문장을 삭제하려면 다음 세 조건을 모두 만족해야 합니다.

    1. ``이미지``라는 단어를 포함한다.
    2. 첨부 또는 업로드를 뜻하는 표현을 포함한다.
    3. 질문·전송·검색 재시도를 뜻하는 표현을 포함한다.

    세 조건을 함께 검사하면 이미지에 관해 정상적으로 설명하는 문장을
    재첨부 요청으로 잘못 판단할 가능성을 줄일 수 있습니다.

    Args:
        answer: 모델이 생성한 원본 답변입니다. 방어적으로 ``None``과 같은
            falsy 값도 빈 문자열로 취급합니다.
        has_image: 현재 사용자 요청에 이미지가 이미 포함되었는지 여부입니다.

    Returns:
        불필요한 재첨부 요청을 제거한 답변입니다. 제거 후 남는 문장이 없으면
        사용자가 다음 행동을 취할 수 있도록 일반적인 재질문 안내를 반환합니다.
    """
    # 앞뒤 공백을 제거해 빈 답변을 일관되게 처리하고, 이후 문장 결합 시
    # 불필요한 공백이 중복되는 것을 방지합니다.
    normalized_answer = (answer or "").strip()

    # 이미지가 없는 요청에는 재첨부 안내가 유효할 수 있으므로 수정하지 않습니다.
    # 답변이 비어 있다면 더 처리할 문장도 없으므로 즉시 반환합니다.
    if not has_image or not normalized_answer:
        return normalized_answer

    # 줄바꿈으로 나열된 문장과 일반적인 문장부호로 끝난 문장을 각각 검사할 수
    # 있도록 답변을 작은 단위로 분리합니다.
    sentences = _SENTENCE_BOUNDARY.split(normalized_answer)

    # 공백뿐인 조각은 버리고, 이미지 재첨부/재시도를 요구한다고 판단되는
    # 문장만 제외합니다. 세 가지 조건은 오탐을 줄이기 위해 모두 충족해야 합니다.
    kept_sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
        and not (
            "이미지" in sentence
            and any(term in sentence for term in _UPLOAD_TERMS)
            and any(term in sentence for term in _RETRY_TERMS)
        )
    ]

    # 필터링을 통과한 문장은 한 칸의 공백으로 이어 붙여 정돈된 답변을 만듭니다.
    if kept_sentences:
        return " ".join(kept_sentences)

    # 모든 문장이 제거된 경우 빈 응답 대신 이미지 재첨부를 요구하지 않는
    # 후속 안내를 제공하여 대화가 막히지 않도록 합니다.
    return "검색 조건을 조금 바꿔 다시 질문해 주시면 더 정확하게 찾아드릴 수 있습니다."

from pydantic import BaseModel, Field, field_validator
from typing import List, Any
from pydantic import ConfigDict

# 1. 응답 데이터 구조 정의
class SearchResultItem(BaseModel):
    # 명시되지 않은 필드(예: DB에서 조인해 온 다양한 컬럼들)도 모두 허용
    # extra='allow' 옵션을 켜 두었기 때문에 정의하지 않은 필드(예: brand, lprice, category1 등)도 에러 없이 그대로 통과
    model_config = ConfigDict(extra='allow')

    # 공통 혹은 빈번한 필드만 필수로 지정
    id: Any = None
    title: Any = None
    description: Any = None
    image_url: Any = None
    score: Any = None

    # ── [Score 변환기] ──
    # Qdrant 검색 결과로 넘어오는 score 값은 일반 Python float가 아니라
    # Numpy 객체인 numpy.float64 타입일 수 있습니다. (이 경우 FastAPI가 JSON으로 못 바꿔서 500 에러 발생)
    # mode='before' 옵션: Pydantic이 엄격한 타입 검증을 시작하기 '전'에 이 함수를 먼저 가로채어 실행하겠다는 의미입니다.
    @field_validator('score', mode='before')
    @classmethod
    def parse_score(cls, v):
        # 만약 들어온 값(v)에 item() 메서드가 있다면 (즉, Numpy 스칼라 타입이라면)
        if hasattr(v, 'item'):
            # item()을 호출하여 순수한 Python float 값으로 변환해서 돌려줍니다.
            return v.item()
        # Numpy 타입이 아니라 이미 일반 float면, 건드리지 않고 원래 값 그대로 통과시킵니다.
        return v

# 2. 최종 API 응답 전체 구조 정의
class SearchResponse(BaseModel):
    results: List[SearchResultItem]  # 여러 개의 상품 검색 결과 리스트
    answer: str                      # LLM이 생성한 안내 메시지 텍스트

    # ── [Answer 변환기] ──
    # 환경(Gemini 등)에 따라 LangChain의 응답이 단순 문자열이 아니라
    # AIMessage(content="안녕하세요...", ...) 같은 복잡한 객체 형태로 넘어올 때가 있습니다.
    @field_validator('answer', mode='before')
    @classmethod
    def parse_answer(cls, v):
        # 들어온 값(v)이 AIMessage 객체라서 내부에 'content' 속성을 가지고 있다면
        if hasattr(v, 'content'):
            # 객체 전체를 JSON으로 바꾸려다 에러나지 않게, 핵심 텍스트 내용만 쏙 뽑아줍니다.
            return v.content
        # 객체가 아니라면 안전하게 문자열(str)로 강제 형변환합니다.
        # (만약 v가 None 등 비어있는 값이면 안전하게 빈 문자열 ""을 반환합니다)
        return str(v) if v else ""
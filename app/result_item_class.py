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

    # Pydantic v2 방식의 validator로 Numpy float를 파이썬 float로 변환
    @field_validator('score', mode='before')
    @classmethod
    def parse_score(cls, v):
        if hasattr(v, 'item'):
            return v.item()
        return v

class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    answer: str
    @field_validator('answer', mode='before')
    @classmethod
    def parse_answer(cls, v):
        if hasattr(v, 'content'):
            return v.content
        return str(v) if v else ""
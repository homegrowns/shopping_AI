from typing import Any, List, Optional

from langgraph.graph import MessagesState


# 그래프 내부에서 사용하는 전체 상태
class AgentState(MessagesState):
    question: Optional[str]
    label_text: Optional[str]

    context: str
    answer: str
    retry_num: int

    query_vector: Optional[List[float]]
    search_results: List[Any]
    is_image_collection: bool


# graph.invoke()로 처음 전달하는 입력 상태
class InputState(MessagesState):
    question: Optional[str]
    label_text: Optional[str]

    query_vector: Optional[List[float]]
    is_image_collection: bool

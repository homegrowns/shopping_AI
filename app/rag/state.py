from typing import List, Optional, Any
from langgraph.graph import MessagesState, StateGraph

# 1. 그래프 내부에서 굴러갈 전체 상태(State)
class AgentState(MessagesState):
    question: Optional[str] = None
    context: str
    answer: str
    retry_num: int
    query_vector: List[float] = None      
    search_results: List[Any] 

# 2. 처음에 graph.invoke()로 입력받을 데이터의 상태(State)
class InputState(MessagesState):
    question: Optional[str] = None
    query_vector: List[float] = None
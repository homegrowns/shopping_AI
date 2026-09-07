import re
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from app.rag.edges import check_hallucinations, decide_to_generate
from app.rag.nodes import (
    chatbot,
    context_organizer,
    generate,
    qdrant_search,
    route_tools,
    small_talk,
    transform_query,
    transform_sql_query,
)
from app.rag.state import AgentState, InputState

graph_builder = StateGraph(AgentState, input_schema=InputState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("qdrant_search", qdrant_search)
graph_builder.add_node("small_talk", small_talk)

graph_builder.add_edge(START, "chatbot")
# tools_condition 대신 route_tools 사용!
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {
        "tools": "qdrant_search",
        "small_talk": "small_talk",
        END: END,
    },
)

graph_builder.add_node("context_organizer", context_organizer)
graph_builder.add_node("transform_query", transform_query)
graph_builder.add_node("generate", generate)


graph_builder.add_edge("qdrant_search", "context_organizer")
graph_builder.add_edge("context_organizer", "generate")
# graph_builder.add_conditional_edges(
#     "context_organizer",
#     decide_to_generate,
#     {
#         "transform_query": "transform_query",
#         # "transform_sql_query": "transform_sql_query",
#         "generate": "generate",
#     },
# )

graph_builder.add_edge("small_talk", END)

# graph_builder.add_edge("transform_query", "qdrant_search")

graph_builder.add_conditional_edges(
    "generate",
    check_hallucinations,
    {"not supported": "generate", "support": END},
)


graph = graph_builder.compile()


def start_agent(
    query_text: str | None = None,
    label_text: str | None = None,
    query_vector: list | None = None,
    is_image_collection: bool = True,
):
    label = (label_text or "").strip()

    # 1. 텍스트가 특수기호로만 이루어져 있는지 확인
    is_meaningless = False
    if query_text:
        cleaned_text = re.sub(r"[^\w\sㄱ-ㅎ가-힣]", "", query_text).strip()
        if not cleaned_text:
            is_meaningless = True
    # 2. 이미지만 있는 경우
    if query_text is None:
        safe_message = "이 이미지와 유사한 상품을 찾아주세요." + label

    # 3. [핵심] 의미 없는 입력인 경우 -> LLM(그래프)을 아예 호출하지 않고 즉시 종료!
    elif is_meaningless:
        print(f"-----MEANINGLESS INPUT DETECTED: {query_text} -----")
        # Graph를 타지 않고, LangGraph가 반환할 최종 State(딕셔너리) 형태를 직접 만들어서 즉시 반환합니다.
        return {
            "messages": [
                HumanMessage(content=query_text),
                AIMessage(
                    content="무엇을 도와드릴까요? 원하시는 상품명이나 특징을 구체적으로 입력해주세요. "
                ),
            ],
            # 추가된 부분: 빈 결과값과 빈 컨텍스트를 명시적으로 넘겨줍니다.
            "context": "",
            "search_results": [
                {
                    "answer": "무엇을 도와드릴까요? 원하시는 상품명이나 특징을 구체적으로 입력해주세요. "
                }
            ],  # 결과가 없으므로 빈 리스트 반환
        }

    # 4. 정상적인 텍스트인 경우
    else:
        safe_message = query_text

    # 유효한 입력일 때만 Graph(LLM) 호출
    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=safe_message)],
            "question": query_text,
            "label_text": label or None,
            "query_vector": query_vector,
            "is_image_collection": is_image_collection,
        }
    )
    return final_state

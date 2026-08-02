import re
from typing import Optional
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph, START, END

from app.rag.state import AgentState, InputState
from app.rag.nodes import chatbot, qdrant_search, sql_query_generate, route_tools, context_organizer, generate, transform_query, transform_sql_query, small_talk
from app.rag.edges import decide_to_generate, check_hallucinations


graph_builder = StateGraph(AgentState, input_schema=InputState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("qdrant_search", qdrant_search)
# graph_builder.add_node("sqllite", sql_query_generate)
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
    }
)

graph_builder.add_node("context_organizer", context_organizer)
graph_builder.add_node("transform_query", transform_query)
# graph_builder.add_node("transform_sql_query", transform_sql_query)
graph_builder.add_node("generate", generate)


graph_builder.add_edge("qdrant_search", "context_organizer")
graph_builder.add_conditional_edges(
    "context_organizer",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        # "transform_sql_query": "transform_sql_query",
        "generate": "generate",
    },
)

graph_builder.add_edge("small_talk", END)

graph_builder.add_edge("transform_query", "qdrant_search")
# graph_builder.add_edge("transform_sql_query", "sqllite")
graph_builder.add_conditional_edges(
    "generate",
    check_hallucinations,
    {
        "not supported": "generate",
        "support": END
    },
)


graph = graph_builder.compile()

# def start_agent(query_text: Optional[str], query_vector: list):
#     # 만약 사용자가 이미지만 올렸다면(query_text가 None이라면) 기본 메시지로 치환
#     safe_message = query_text if query_text else "이 이미지와 유사한 상품을 찾아주세요."
#     try:
#         # 그래프 이미지 저장 로직
#         png_bytes = graph.get_graph().draw_mermaid_png()
#         with open("/home/liam/shopping_ai/shopping_assistant/app/rag/graph.png", "wb") as f:
#             f.write(png_bytes)
#     except Exception:
#         pass
#     # 이제 텍스트 메시지와 벡터 데이터를 각각 분리해서 State에 꽂아 넣습니다.
#     final_state = graph.invoke(
#         {
#             "messages": [HumanMessage(content=safe_message)],
#             "question": query_text,
#             "query_vector": query_vector,
#         }
#     )
#     return final_state



def start_agent(query_text: Optional[str], query_vector: list):
    # 1. 텍스트가 특수기호로만 이루어져 있는지 확인
    is_meaningless = False
    if query_text:
        cleaned_text = re.sub(r'[^\w\sㄱ-ㅎ가-힣]', '', query_text).strip()
        if not cleaned_text:
            is_meaningless = True
    # 2. 이미지만 있는 경우
    if query_text is None:
        safe_message = "이 이미지와 유사한 상품을 찾아주세요."
        
    # 3. [핵심] 의미 없는 입력인 경우 -> LLM(그래프)을 아예 호출하지 않고 즉시 종료!
    elif is_meaningless:
        print(f"-----MEANINGLESS INPUT DETECTED: {query_text} -----")
        # Graph를 타지 않고, LangGraph가 반환할 최종 State(딕셔너리) 형태를 직접 만들어서 즉시 반환합니다.
        return {
            "messages": [
                HumanMessage(content=query_text),
                AIMessage(content="무엇을 도와드릴까요? 원하시는 상품명이나 특징을 구체적으로 입력해주세요. ")
            ],
            # 추가된 부분: 빈 결과값과 빈 컨텍스트를 명시적으로 넘겨줍니다.
            "context": "",
            "search_results": [{'answer': '무엇을 도와드릴까요? 원하시는 상품명이나 특징을 구체적으로 입력해주세요. '}]  # 결과가 없으므로 빈 리스트 반환
        }
        
    # 4. 정상적인 텍스트인 경우
    else:
        safe_message = query_text
    try:
        # 그래프 이미지 저장 로직
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("/home/liam/shopping_ai/shopping_assistant/app/rag/graph.png", "wb") as f:
            f.write(png_bytes)
    except Exception:
        pass
        
    # 유효한 입력일 때만 Graph(LLM) 호출
    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=safe_message)],
            "question": query_text,
            "query_vector": query_vector,
        }
    )
    return final_state
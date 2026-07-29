from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from typing import List, Optional

load_dotenv()

from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph, START, END

from app.rag.state import AgentState, InputState
from app.rag.nodes import chatbot, qdrant_search, sql_query_generate, route_tools, context_organizer, generate, transform_query, transform_sql_query
from app.rag.edges import decide_to_generate, check_hallucinations


graph_builder = StateGraph(AgentState, input_schema=InputState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("qdrant_search", qdrant_search)
# graph_builder.add_node("sqllite", sql_query_generate)

graph_builder.add_edge(START, "chatbot")
# tools_condition 대신 route_tools 사용!
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {
        "tools": "qdrant_search",
        # "sql_tool": "sqllite",
        END: END,
    }
)

graph_builder.add_node("context_organizer", context_organizer)
graph_builder.add_node("transform_query", transform_query)
# graph_builder.add_node("transform_sql_query", transform_sql_query)
graph_builder.add_node("generate", generate)

graph_builder.add_edge("qdrant_search", "context_organizer")
# graph_builder.add_edge("sqllite", "context_organizer")
graph_builder.add_conditional_edges(
    "context_organizer",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        # "transform_sql_query": "transform_sql_query",
        "generate": "generate",
    },
)
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

# if __name__ == "__main__":

# def start_agent(query: str):
#     try:
#         png_bytes = graph.get_graph().draw_mermaid_png()
#         with open("/home/liam/shopping_ai/shopping_assistant/app/rag/graph.png", "wb") as f:
#             f.write(png_bytes)
#     except Exception:
#         pass

#     final_state = graph.invoke(
#         {
#             "messages": [
#                 HumanMessage(content=query)
#             ]
#         }
#     )
#     return final_state


def start_agent(query_text: Optional[str], query_vector: list,  eng_text: bool = False):
    # 만약 사용자가 이미지만 올렸다면(query_text가 None이라면) 기본 메시지로 치환
    safe_message = query_text if query_text else "이 이미지와 유사한 상품을 찾아주세요."
    try:
        # 그래프 이미지 저장 로직
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("/home/liam/shopping_ai/shopping_assistant/app/rag/graph.png", "wb") as f:
            f.write(png_bytes)
    except Exception:
        pass
    # 이제 텍스트 메시지와 벡터 데이터를 각각 분리해서 State에 꽂아 넣습니다.
    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=safe_message)],
            "question": query_text,
            "query_vector": query_vector,
            "eng_text": eng_text,
        }
    )
    return final_state
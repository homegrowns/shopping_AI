import re
from typing import Generator, List, Optional

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


async def start_agent(
    query_text: str | None = None,
    label_text: str | None = None,
    query_vector: list | None = None,
    is_image_collection: bool = True,
) -> Generator[dict, None, None]:
    """
    graph.stream()으로 토큰 단위 스트리밍 + 최종 state 수집을 한 번에 처리합니다.
    LLM을 2번 호출할 필요 없이 하나의 stream에서 모든 데이터를 수집합니다.

    Yields:
        dict 형태의 이벤트:
        - {"type": "token", "content": "..."} — generate 노드의 LLM 토큰
        - {"type": "done", "search_results": [...], "answer": "..."} — 최종 결과
    """
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
        fallback_answer = "무엇을 도와드릴까요? 원하시는 상품명이나 특징을 구체적으로 입력해주세요. "
        yield {"type": "token", "content": fallback_answer}
        yield {"type": "done", "search_results": [], "answer": fallback_answer}
        return

    # 4. 정상적인 텍스트인 경우
    else:
        safe_message = query_text

    try:
        # 그래프 이미지 저장 로직
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open(
            "/home/liam/shopping_ai/shopping_assistant/app/rag/graph.png", "wb"
        ) as f:
            f.write(png_bytes)
    except Exception:
        pass

    # ── 핵심: stream_mode를 리스트로 지정하면 토큰 + state 업데이트를 동시에 받음 ──
    search_results = []
    answer_chunks = []

    async for event in graph.astream(
        {
            "messages": [HumanMessage(content=safe_message)],
            "question": query_text,
            "label_text": label or None,
            "query_vector": query_vector,
            "is_image_collection": is_image_collection,
        },
        stream_mode=["messages", "updates", "custom"],
    ):
        kind, data = event

        if kind == "custom":
            # 노드 내부에서 writer()로 보낸 데이터
            print(f"상태 메세지 : {data}")
            yield data

        elif kind == "updates":
            # data = {node_name: node_output_dict}
            for node_name, node_output in data.items():

                if not isinstance(node_output, dict):
                    continue
                # 검색 결과 저장
                if "search_results" in node_output:
                    search_results = node_output["search_results"]

                # generate 노드가 완료되면
                # node_output 자체가 generate 노드의 반환값이다.
                if node_name == "generate":

                    answer = node_output.get("answer")
                    answer_chunks.append(answer)
                    if answer:
                        yield {
                            "type": "token",
                            "content": answer,
                        }
        elif kind == "messages":
            msg, metadata = data
            node_name = metadata.get("langgraph_node", "")
    
            # small_talk 노드의 응답도 스트리밍
            if node_name == "small_talk" and msg.content:
                answer_chunks.append(msg.content)
                yield {"type": "token", "content": msg.content}

    # 모든 스트리밍이 끝난 뒤 최종 결과를 한 번에 전송
    yield {
        "type": "done",
        "search_results": search_results,
        "answer": "".join(answer_chunks),
    }


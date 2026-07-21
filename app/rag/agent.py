from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph, MessagesState, START, END

from app.rag.state import AgentState
from app.rag.nodes import chatbot, retrieve, sql_query_generate, route_tools, context_organizer, generate, transform_query
from app.rag.edges import decide_to_generate, check_hallucinations


graph_builder = StateGraph(AgentState, input_schema=MessagesState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("retriever", retrieve)
graph_builder.add_node("sqllite", sql_query_generate)

graph_builder.add_edge(START, "chatbot")
# tools_condition 대신 route_tools 사용!
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {
        "tools": "retriever",
        "sql_tool": "sqllite",
        END: END,
    }
)

graph_builder.add_node("context_organizer", context_organizer)
graph_builder.add_node("transform_query", transform_query)
graph_builder.add_node("generate", generate)

graph_builder.add_edge("retriever", "context_organizer")
graph_builder.add_edge("sqllite", "context_organizer")
graph_builder.add_conditional_edges(
    "context_organizer",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
graph_builder.add_edge("transform_query", "retriever")
graph_builder.add_edge("transform_query", "sqllite")
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

def start_agent(query: str):
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("/home/liam/shopping_ai/shopping_assistant/app/rag/graph.png", "wb") as f:
            f.write(png_bytes)
    except Exception:
        pass

    final_state = graph.invoke(
        {
            "messages": [
                HumanMessage(content=query)
            ]
        }
    )
    return final_state

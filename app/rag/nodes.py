import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from app.rag.sql_tool  import execute_sql_query, get_table_schema
from app.rag.retriever import retriever, retriever_tool
from app.rag.state import AgentState


if os.getenv("ENV") == "prod":
    from langchain_aws import ChatBedrock
    llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet...")
    print("(nodes.py) LLM: ", "prod")

elif os.getenv("ENV") == "local":
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    print("(nodes.py) LLM: ", "local")

elif os.getenv("ENV") == "dev":
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.1")
    print("(nodes.py)LLM: ", "dev")

llm_with_tools = llm.bind_tools([retriever_tool, execute_sql_query, get_table_schema])


system_prompt = """당신은 쇼핑몰 AI 어시스턴트입니다.
[문서 검색]
- 맞춤법, 규정 관련 질문 → pdf_search 도구 사용
[상품 DB 검색]
- 상품 조회, 가격, 개수 질문 → 아래 규칙 따르기
  1. 먼저 get_table_schema로 스키마 확인
  2. execute_sql_query로 SELECT만 실행
  3. DROP/DELETE/UPDATE/INSERT 절대 금지
  4. 색상 같은 컬럼 없으면 title, description에 LIKE 검색
[일반 질문]
- 도구 없이 직접 답변
[중요 사항]
- 상품 관련 질문은 절대 pdf_search를 사용하지 마세요!
"""

def chatbot(state: AgentState):
    """
    검색(Retriever) 도구를 바인딩 한 LLM 모델에 현재 메시지 상태를 입력하여 응답을 생성합니다.
    질문이 주어지면 검색 도구를 도구호출 하거나 일반 답변하며 종료할지 결정할 수 있습니다.
    """
    print("----- [CHATBOT] -----")
    # system_prompt를 MessagesState에 추가하기 위해 AI Message로 변환
    system_message = AIMessage(content=system_prompt)
    
    # state에 system 메시지를 먼저 추가하고 나머지 메시지들을 뒤에 이어 붙입니다.
    messages = [system_message] + state["messages"]
    
    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        "question": messages[-1].content
    }

def retrieve(state: AgentState):
    """
    현재 질문을 기반으로 관련 문서를 검색합니다.
    """
    print("----- [RETRIEVER] -----")
    question = state["question"]
    relevant_doc = retriever.invoke(question) # [ 1 ]
    context = ""
    for doc in relevant_doc: # [ 2 ]
        context += f"Page {doc.metadata['page']+1}: {doc.page_content}\n"


    # Tool 호출에 대한 응답 메시지(검색 결과) 생성
    last_message = state["messages"][-1]

    if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
        tool_call_id = last_message.tool_calls[0]['id']
        tool_message = ToolMessage( # [ 3 ]
            content=context,
            name="retriever",
            tool_call_id=tool_call_id
        )
        return {"messages": [tool_message], "context": context} # [ 4 ]
    else:
        return {"messages": [AIMessage(content=context)], "context": context}

def route_tools(state: AgentState):
    last_message = state["messages"][-1]
    
    # 도구 호출 없으면 종료
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END
    
    # 어떤 도구인지 확인
    tool_name = last_message.tool_calls[0]["name"]
    
    if tool_name == "pdf_search":
        print("----- [ROUTE TOOLS PDF SEARCH] -----")
        return "tools"        # → retriever
    elif tool_name in ["execute_sql_query", "get_table_schema"]:
        print("----- [ROUTE TOOLS SQL QUERY] -----")
        return "sql_tool"     # → sqllite
    else:
        print("----- [END] -----")
        return END

def sql_query_generate(state: AgentState):
    """
    현재 질문을 기반으로 관련 sqllite를 검색합니다.
    """
    print("----- [SQL QUERY GENERATE] -----")
    # Tool 호출에 대한 응답 메시지(검색 결과) 생성
    last_message = state["messages"][-1]

    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": [AIMessage(content="SQL 호출 정보 없음")]}
    
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    # 도구 실행
    if tool_name == "execute_sql_query":
        context = execute_sql_query.invoke(tool_args)
    elif tool_name == "get_table_schema":
        context = get_table_schema.invoke(tool_args)
    
    tool_message = ToolMessage(
        content=str(context),
        name=tool_name,
        tool_call_id=tool_call["id"]
    )
    return {"messages": [tool_message], "context": str(context)}

def context_organizer(state: AgentState):
    """
    검색된 결과를 정리합니다.
    """
    print("----- [CONTEXT ORGANIZER] -----")
    context = state["context"]

    context_organizer_prompt = ChatPromptTemplate.from_messages(
        [
            (   "system",
                """당신은 검색증강생성(RAG)을 위한 검색문서 및 쿼리 결과를 정리하는 전문가입니다.
                아래의 검색된 결과를 확인하고, LLM이 해당 문서를 정리된 형태로 참고할 수 있도록
                문서의 불필요한 공백 등을 삭제하거나 정렬을 다시하여 정리된 형태로 반환해주세요.
                내용을 삭제하는 것을 최소로 합니다. 페이지 번호 정보를 절대 삭제하지 마세요.
                SQL QUERY GENERATE사용시 쿼리를 보여주지말고 결과만 보여주세요"""
            ),
            (
                "user",
                """
                검색 결과: {context}
                """,
            ),
        ]
    )

    context_organizer = context_organizer_prompt | llm
    organized_context = context_organizer.invoke({"context": context})

    return {"context": organized_context.content, "messages": [AIMessage(organized_context.content)]}



def transform_query(state: AgentState):
    """
    더 나은 질문을 생성하기 위해 쿼리를 변환합니다.

    Args:
        state (dict): 현재 그래프 상태

    Returns:
        state (dict): 재구성된 질문으로 question 키를 업데이트
    """

    print("----- [TRANSFORM QUERY] -----")
    question = state["question"]

    system = """
    당신은 질문을 다시 작성하는 전문가입니다. 입력된 질문을 검색에 최적화된 더 나은 버전으로 변환하세요.
    입력을 살펴보고 질문의 핵심적인 의미와 의도를 파악하여 개선된 질문을 만들어주세요."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "user",
                "다음은 초기 질문입니다: \n\n {question} \n 한국어로 개선된 질문을 작성해주세요.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm

    better_question = question_rewriter.invoke({"question": question})
    return {"question": better_question.content, "messages": [better_question], "retry_num": state["retry_num"] + 1 if state.get("retry_num") else 1}


def generate(state: AgentState):
    """
    검색된 문서와 질문을 기반으로 답변을 생성합니다.
    """
    print("----- [GENERATE] -----")
    question = state["question"]
    context = state["context"]

    retry_num = state.get("retry_num", 0)

    if retry_num >= 3: # [ 1 ]
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 검색된 문서를 통해 해결할 수 있는 질문을 추출하는 어시스턴트입니다.
                    사용자가 해결하고자 한 질문이 있었으나 검색 컨텍스트가 충분하지 않은 상황이므로, 주어진 검색 결과 내에서 답변할 수 있는 질문을 새롭게 작성해 나열하세요.
                    사용자에게 질문에 대한 답변을 하지 못함에 양해를 구하고, 다른 질문의 기회와 선택지를 제공하는 친절한 가이드를 하세요.
                    """,
                ),
                (
                    "user",
                    "질문: {question} \n\n검색 결과: {context} \n\n답변:",
                ),
            ]
        )
    else: # [ 2 ]
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 질문-답변 업무를 수행하는 어시스턴트입니다. 검색된 컨텍스트를 사용하여 질문에 답변하세요.
                    답변을 모르는 경우, 모른다고 말하세요.
                    답변은 간결하게 작성하고, 반드시 답변의 출처(페이지 번호)를 함께 명시해주세요.


                    """,
                ),
                (
                    "user",
                    "질문: {question} \n\n검색 결과: {context} \n\n답변:",
                ),
            ]
        )

    rag_chain = rag_prompt | llm
    response = rag_chain.invoke({"question": question, "context": context})
    return {"question": question, "answer": response.content, "messages": [response]}

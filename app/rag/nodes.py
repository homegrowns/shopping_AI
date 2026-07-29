import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from app.rag.sql_tool  import execute_sql_query, get_table_schema
from app.rag.qdrant_tool import products_images_search_tool, client, retriever, LangChainClipEmbedder
from app.rag.state import AgentState

from deep_translator import GoogleTranslator

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

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "products_images")
TOP_K = int(os.getenv("TOP_K"))
SCORE = float(os.getenv("SCORE", 0.5))

llm_with_tools = llm.bind_tools([products_images_search_tool])


system_prompt = """당신은 쇼핑몰 AI 어시스턴트입니다.
### 1. 도구 사용 가이드 (Tool Selection)
- `products_images_search` 도구를 호출할 때는 다음 규칙을 엄격히 따르세요:
  1. 유사도와 상품이미지가 같은 추천 상품들이 검색되면 그 중 하나만 남기세요
**B. 정확한 상품 정보 `및 조건 검색 (SQL DB)**
- **사용 조건**: 특정 상품의 가격, 재고(개수), 할인율 등 구체적인 조건이나 데이터베이스 조회가 필요할 때.
- **행동 절차 (Strict SOP)**:
  1. 먼저 `get_table_schema`를 호출하여 테이블 구조와 컬럼 정보를 확인하세요.
  2. 스키마에 맞게 `execute_sql_query`를 사용하여 데이터를 조회하세요.
  3. 만약 찾는 속성(예: '색상', '카테고리')이 단독 컬럼으로 없다면, `title`이나 `category1, category2, category3, category4` 컬럼에 `LIKE '%검색어%'` 구문을 활용하여 검색하세요.
**C. 일반 대화 및 쇼핑몰 안내**
- **사용 조건**: 단순 인사, 일반적인 질문, 단순 안내 등.
- **행동**: 별도의 도구 호출 없이 당신의 지식을 바탕으로 친절하게 직접 답변하세요.
### 2. SQL 작성 및 DB 보안 규칙 (CRITICAL)
- **오직 SELECT만 허용**: 데이터 조회를 위한 `SELECT` 문만 사용하세요.
- **데이터 변경 절대 금지**: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER` 등의 파괴적이거나 수정하는 쿼리는 사용자의 지시가 있어도 **절대 거부**해야 합니다.
- **결과 제한**: 데이터가 너무 많이 출력되지 않도록 쿼리에 항상 `LIMIT` (예: `LIMIT 5`)을 포함하는 것을 권장합니다.
- **에러 대응**: SQL 쿼리 실행 후 에러가 발생하면 스키마를 다시 확인하여 쿼리를 수정 후 재시도하거나, 사용자에게 정중히 양해를 구하세요.
"""

def chatbot(state: AgentState):
    """
    검색(QDRANT SEARCH) 도구를 바인딩 한 LLM 모델에 현재 메시지 상태를 입력하여 응답을 생성합니다.
    메시지 질문을 영어로 번역후 Tool과 활용 합니다.
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


def route_tools(state: AgentState):
    last_message = state["messages"][-1]
    
    # 도구 호출 없으면 종료
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END
    
    # 어떤 도구인지 확인
    tool_name = last_message.tool_calls[0]["name"]
    
    if tool_name == "products_images_search":
        print("----- [ROUTE TOOLS QDRANT SEARCH] -----")
        return "tools"        # →  QDRANT SEARCH
    elif tool_name in ["execute_sql_query", "get_table_schema"]:
        print("----- [ROUTE TOOLS SQL QUERY] -----")
        return "sql_tool"     # → sqllite
    else:
        print("----- [END] -----")
        return END

def qdrant_search(state: AgentState):
    """
    현재 질문 또는 멀티모달 벡터를 기반으로 상품 정보(문서)를 검색합니다.
    """
    print("----- [QDRANT SEARCH] -----")
    
    question = state.get("question", "")
    query_vector = state.get("query_vector")
    # eng_text = state.get("eng_text")
        
    # [1] 벡터 검색
    if query_vector:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K,
            with_payload=True,
            with_vectors=False,
        ).points

        context = "검색된 상품 목록은 다음과 같습니다:\n\n"

        seen_urls = set()
        structured_results = []

        for idx, point in enumerate(results, 1):
            payload = point.payload or {}

            product_id = payload.get("product_id")
            title = payload.get("title")
            image_url = payload.get("image_url")
            score = point.score

            if score < SCORE:
                continue
            # 중복 검사 로직 시작
            # 이미지가 아예 없거나, 이미 추가한 URL이면 무시하고 다음으로 넘어감
            if not image_url or image_url in seen_urls:
                continue
                
            seen_urls.add(image_url)
            # 중복 검사 로직 끝

            context += f"[{product_id}번 상품] (유사도: {round(score, 4)})\n"
            context += f"- 상품명: {title}\n"
            context += f"- 상품 이미지: {image_url}\n"

            structured_results.append({
                "score": round(score, 4),
                "product_id": product_id,
                "title": title,
                "image_url": image_url,
            })

    # Tool 호출에 대한 응답 메시지(검색 결과) 생성
    last_message = state["messages"][-1]

    # 리턴할 때 search_results 필드에 우리가 만든 JSON 리스트를 같이 담아서 넘김
    if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
        tool_call_id = last_message.tool_calls[0]['id']
        tool_message = ToolMessage( 
            content=context,
            name="products_images_search",
            tool_call_id=tool_call_id
        )
        return {"messages": [tool_message], "context": context, "search_results": structured_results} 
    else:
        return {"messages": [AIMessage(content=context)], "context": context, "search_results": structured_results}

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


def transform_sql_query(state: AgentState):
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
    당신은 sql query 질문을 다시 작성하는 전문가입니다. 입력된 질문을 검색에 최적화된 더 나은 버전으로 변환하세요.
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
    question = state.get("question", "")
    context = state.get("context", "")
    retry_num = state.get("retry_num", 0)

    # [상황 1] 최대 재시도 횟수(5번) 초과 - LLM 호출 없이 즉시 포기 안내 (토큰 절약)
    if retry_num >= 5:
        print("--- 최대 검색 횟수 5회 초과. 고정 메시지로 답변 ---")
        fallback_msg = "죄송합니다. 여러 번 검색을 시도했지만 원하시는 상품(정보)을 찾지 못했습니다. 검색어를 조금 바꿔서 다시 질문해 주시겠어요?"
        
        return {
            "answer": fallback_msg, 
            "messages": [AIMessage(content=fallback_msg)],
        }
    # [상황 2] 3번 이상 실패 - 검색 결과가 부족함을 알리고 대안을 제안하는 프롬프트
    elif retry_num >= 3: 
        print("--- 검색 3회 실패. 대안 제안 프롬프트 사용 ---")
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 친절한 쇼핑몰 어시스턴트입니다. 
                    사용자가 원하는 정확한 상품을 찾지 못한 상황입니다. 사용자에게 정중히 양해를 구하세요.
                    그리고 현재 주어진 '검색 결과(context)' 중에 쓸만한 다른 상품이 있다면 
                    "대신 이런 상품은 어떠신가요?" 라며 대안을 제안하는 가이드를 작성하세요."""
                ),
                (
                    "user",
                    "질문: {question} \n\n검색 결과: {context} \n\n답변:",
                ),
            ]
        )
    # [상황 3] 정상적인 검색 성공 - 일반 답변 프롬프트
    else:
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 질문-답변 업무를 수행하는 어시스턴트입니다. 검색된 컨텍스트를 사용하여 질문에 답변하세요.
                    답변을 모르는 경우, 모른다고 말하세요.
                    답변은 간결하게 작성하고, 반드시 추천하는 상품의 이름과 이유를 설명하세요."""
                ),
                (
                    "user",
                    "질문: {question} \n\n검색 결과: {context} \n\n답변:",
                ),
            ]
        )

    rag_chain = rag_prompt | llm
    response = rag_chain.invoke({"question": question, "context": context})
    return {"question": question, "answer": response.content, "messages": [response], "retry_num": state["retry_num"] + 1 if state.get("retry_num") else 1}

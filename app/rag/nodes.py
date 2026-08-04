import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, SystemMessage
from app.rag.sql_tool  import execute_sql_query, get_table_schema
from app.rag.qdrant_tool import products_images_search_tool, client, retriever, LangChainClipEmbedder
from app.rag.communication_tool import handle_small_talk
from app.rag.state import AgentState
from langgraph.graph import END

from deep_translator import GoogleTranslator

if os.getenv("ENV") == "prod":
    from langchain_aws import ChatBedrock
    llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet...")
    print("(nodes.py) LLM: ", "prod anthropic.claude-3-5-sonnet")

elif os.getenv("ENV") == "local":
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    print("(nodes.py) LLM: ", "local gemini-2.5-flash")

elif os.getenv("ENV") == "dev":
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.1")
    print("(nodes.py)LLM: ", "dev llama3.1")

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "products_images")
TOP_K = int(os.getenv("TOP_K"))
SCORE = float(os.getenv("SCORE", 0.4))

llm_with_tools = llm.bind_tools([products_images_search_tool, handle_small_talk])


system_prompt = """
당신은 쇼핑몰 AI 어시스턴트입니다.

## 이름
- 조디

## 역할
- 상품 추천 및 상품 관련 질문에 답변합니다.
- 일반적인 대화도 자연스럽게 응답합니다.
- 사용자가 "안녕" "hello"등 일상적인 인사를 건네면, 도구를 절대 사용하지 말고 "안녕하세요! 무엇을 도와드릴까요?"라고 답변하세요

## 도구 사용 규칙

### products_images_search
다음 경우 반드시 호출하세요.
- 상품 추천 요청
- 상품 이름 (예시: 야구모자)
- 비슷한 상품 찾기
- 코디 추천
- 이미지 기반 상품 추천
- 특정 상품을 찾는 요청
- 관련 상품 카테고리 질문 안내

검색 결과를 사용할 때는 다음을 지키세요.
- 동일한 상품 또는 동일한 이미지의 중복 추천은 제거합니다.
- 검색 결과를 기반으로만 답변합니다.
- 검색 결과가 없으면 없다고 안내합니다.


## 다음과 같은 경우에는 도구를 호출하지 않습니다.
- 쇼핑과 관계없는 질문
- 상품의 총 개수
- 민감한 기술 질문
- 상품의 종류가 명시 되지않은 경우

## 일반 대화(handle_small_talk)
다음 경우 반드시 호출하세요
- 인사
- 어색한 문장의 질문 (예시: Soccer player loss recommendation)
- 완성 되지 않은 문장(예시: grey)
- 상품과 다른주제 대화

## 답변 규칙
- 답변은 간결하고 질문에 맞게 작성합니다.
- 이 이미지와 유사한 상품을 찾아주세요. 라는 인풋메세지가 있으면 "올리신 사진"과 "유사한 상품"이라고 언급하세요
- description을 잘보고 추천이유를 고객이 이해하기 쉽게 잘 설명합니다.
- 검색 결과에 없는 정보는 추측하지 않습니다. (예시: 회색 상품이 있습니다.)
- 이전 질문과 연관지어서 답변하지마세요.
- 상품의 총 개수 같은 질문은 모른다고 하세요
- 상품이나 상품추천 질문아니면 답변하지말고 상품관련 질문만 해달라고 하세요
- IT 기술 질문 무시 예) 파이썬, 랭체인 등등
- 유사도 0.8 이상의 같은 종류상품이 아니면 같은상품 아니라고 말하고 최대한 비슷한 상품을 추천했다고 말합니다.
"""

def chatbot(state: AgentState):
    """
    검색(QDRANT SEARCH) 도구를 바인딩 한 LLM 모델에 현재 메시지 상태를 입력하여 응답을 생성합니다.
    질문이 주어지면 검색 도구를 도구호출 하거나 일반 답변하며 종료할지 결정할 수 있습니다.
    """
    print("----- [CHATBOT] -----")
    # system_prompt를 MessagesState에 추가하기 위해 AI Message로 변환
    # system_message = AIMessage(content=system_prompt)
    system_message = SystemMessage(content=system_prompt)
    # state에 system 메시지를 먼저 추가하고 나머지 메시지들을 뒤에 이어 붙입니다.
    messages = [system_message] + state["messages"]
    
    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
        # 기존: messages[-1].content
        # state["messages"]의 마지막 값을 꺼내는 것이 더 안전할 수 있습니다.
        "question": state["messages"][-1].content 
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
    elif tool_name == "handle_small_talk":
        print("----- [ROUTE small_talk] -----")
        # 핵심 수정: 그래프 이미지에 있는 노드 이름과 똑같이 맞춰줍니다!
        return "small_talk" 
    # else:
    #     print("----- [END] -----")
    #     return END

def qdrant_search(state: AgentState):
    """
    현재 질문 또는 멀티모달 벡터를 기반으로 상품 정보(문서)를 검색합니다.
    """
    print("----- [QDRANT SEARCH] -----")
    
    query_vector = state.get("query_vector")
        
    # [1] 벡터 검색
    if query_vector:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K,
            with_payload=True,
            with_vectors=False,
        ).points

        seen_ids = set()
        structured_results = []
        context = "검색된 상품 목록은 다음과 같습니다:\n"

        for idx, point in enumerate(results, 1):
            payload = point.payload or {}

            product_id = payload.get("product_id")
            # title = payload.get("title")
            description = payload.get("description")
            image_url = payload.get("image_url")
            score = point.score

            if score < SCORE:
                continue
            # 중복 검사 로직 시작
            # 이미지가 아예 없거나, 이미 추가한 URL이면 무시하고 다음으로 넘어감
            if not product_id or product_id in seen_ids:
                continue
                
            seen_ids.add(product_id)
            # 중복 검사 로직 끝

            context += f"[{product_id}번 상품] (유사도: {round(score, 4)})\n"
            # context += f"- 상품명: {title}\n"
            context += f"- 상품 설명: {description}\n"
            context += f"- 상품 이미지: {image_url}\n"

            structured_results.append({
                "score": round(score, 4),
                "product_id": product_id,
                "description": description,
                # "title": title,
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
        return {"messages": [tool_message], "context": context, "search_results": structured_results, "answer": context} 
    else:
        return {"messages": [AIMessage(content=context)], "context": context, "search_results": structured_results, "answer": context}

def small_talk(state: AgentState):
    """
    일상 대화 도구가 호출되었을 때 실행되는 노드입니다.
    """
    print("----- [HANDLE SMALL TALK NODE] -----")
    
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
        tool_call = last_message.tool_calls[0]
        tool_call_id = tool_call['id']
        
        # LLM이 도구를 호출하면서 작성한 'response' 파라미터 값을 꺼냅니다.
        args = tool_call.get('args', {})
        
        # 만약 LLM이 파라미터를 깜빡하고 안 넘겼을 경우를 대비해 기본값을 설정해 줍니다.
        content = args.get('response', "안녕하세요! 쇼핑 어시스턴트입니다. 무엇을 도와드릴까요?")
        
        tool_message = ToolMessage( 
            content=content,
            name="handle_small_talk",
            tool_call_id=tool_call_id
        )
        
        # 검색 결과가 없으므로 search_results는 빈 리스트([])로 넘겨 프론트엔드 에러를 방지합니다.
        return {
            "messages": [tool_message], 
            "context": content, 
            
            # 2.  프론트엔드가 화면에 띄울 수 있게 여기서 "answer"에 직접 대답을 넣어줍니다!
            "answer": content,  
            
            # 3. 엑스박스 방지를 위해 빈 배열을 줍니다.
            "search_results": []
        }
    else:
        # 혹시 도구 호출 정보가 없는 예외 상황을 위한 안전장치
        return {
            "messages": [AIMessage(content="무엇을 도와드릴까요? 원하시는 상품을 말씀해주세요.")], 
            "context": "", 
            "search_results": []
        }

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
                내용을 추가하거나 삭제하지 마세요. 페이지 번호 정보를 절대 삭제하지 마세요.
                [중요사항!] context에 임의로 수정사항을 추가하지말고, 있는 그대로 전달해주세요.
                """
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
    검색된 문서 및 질문을 기반으로 답변을 생성합니다.
    """
    print("----- [GENERATE] -----")
    question = state.get("question", "")
    context = state.get("context", "")
    retry_num = state.get("retry_num", 0)

    # [상황 1] 최대 재시도 횟수(5번) 초과 - LLM 호출 없이 즉시 포기 안내 (토큰 절약)
    if retry_num >= 4:
        print("--- 최대 검색 횟수 4회 초과. 고정 메시지로 답변 ---")
        fallback_msg = "죄송합니다. 여러 번 검색을 시도하고 상품을 찾았으나 만족스러운 검색 결과를 찾지 못했습니다. 검색어를 조금 바꿔서 다시 질문해 주시겠어요?"
        
        return {
            "answer": fallback_msg, 
            "messages": [AIMessage(content=fallback_msg)],
        }
    # [상황 2] 3번 이상 실패 - 검색 결과가 부족함을 알리고 대안을 제안하는 프롬프트
    elif retry_num >= 2: 
        print("--- 검색 실패. 대안 제안 프롬프트 사용 ---")
        rag_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 센스 있고 친절한 쇼핑몰 어시스턴트입니다. 
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
                    """당신은 센스 있고 친절한 쇼핑몰 어시스턴트입니다. 
                    
                    [절대 지켜야 할 답변 규칙]
                    1. 환각 금지: 절대로 가상의 상품명, 가격, 색상을 지어내지 마세요.
                    2. 반말금지: 항상 존댓말을 사용하세요.
                    3. 목록 나열 금지: 검색 결과(context)에 있는 상품 정보를 줄글이나 번호 매기기(1, 2, 3...)로 길게 나열하지 마세요. (사용자 화면 하단에 상품 카드가 자동으로 따로 표시됩니다.)
                    4. 간결한 안내: 정확한 상품이 없을 경우, 사용자에게 정중히 양해를 구하고 "대신 아래에 추천해 드리는 비슷한 상품들을 확인해 보세요!"라는 뉘앙스로 1~2문장 이내의 짧고 친절한 인사말만 작성하세요.
                    5. 불필요한 사족(예: '제가 제공한 검색 결과 중에는~')은 모두 빼고 자연스럽게 대화하듯 말하세요.
                    6. "유사도가 높은 두 개의 상품" 대신 "유사한 상품을 찾았어요"라는 말로 대체 "유사도"라는말 금지 
                    7. 상품 설명 및 description에 질문한 상품이 없으면 찾는 상품 없다고 말하세요
                    """
                ),
                (
                    "user",
                    "질문: {question} \n\n검색 결과: {context} \n\n안내 멘트:",
                ),
            ]
        )

    rag_chain = rag_prompt | llm
    response = rag_chain.invoke({"question": question, "context": context})
    return {"question": question, "answer": response.content, "messages": [response], "retry_num": state["retry_num"] + 1 if state.get("retry_num") else 1}

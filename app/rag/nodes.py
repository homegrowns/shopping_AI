import json
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END

from app.rag.communication_tool import handle_small_talk
from app.rag.qdrant_tool import (
    LangChainClipEmbedder,
    client,
    products_images_search_tool,
    retriever,
)
from app.rag.response_guard import suppress_redundant_image_request
from app.rag.sql_tool import qdrant_to_sql
from app.rag.state import AgentState

if os.getenv("ENV") == "prod":
    from langchain_aws import ChatBedrockConverse  # Converse API 기반 클래스 사용

    print("(nodes.py) LLM: ", "prod anthropic.claude-3-5-sonnet-20240620-v1:0")

    llm = ChatBedrockConverse(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        region_name="ap-northeast-2",  # 서울 리전 명시
        temperature=0.5,
    )

elif os.getenv("ENV") == "stg":
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    print("(nodes.py) LLM: ", "stg gemini-2.5-flash")

elif os.getenv("ENV") == "dev":
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model="llama3.1")
    print("(nodes.py)LLM: ", "dev llama3.1")

COLLECTION_NAME_IMG = os.getenv("QDRANT_COLLECTION_IMG", "products_images")
COLLECTION_NAME_DESC = os.getenv("QDRANT_COLLECTION_DESC", "products_description")
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
- is_image_collection = true 일때
- 상품 추천
- 상품 이름 (예시: 야구모자)
- 비슷한 상품 찾기
- 코디 추천
- 이미지 기반 상품 추천 (메세지: 이미지와 유사한 상품을 찾아)
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

## 일반 대화(handle_small_talk)
다음 경우 반드시 호출하세요
- 인사
- 어색한 문장의 질문 (예시: Soccer player loss recommendation)
- 상품과 다른주제 대화

## 답변 규칙
- 답변은 간결하고 질문에 맞게 작성합니다.
- 이 이미지와 유사한 상품을 찾아주세요. 라는 인풋메세지가 있으면 "올리신 사진"과 "유사한 상품"이라고 언급하세요
- description 및 제품 설명을 잘보고 추천이유를 고객에게 잘 설명합니다.
- 검색 결과가 요청 상품과 종류가 다르면 굳이 결과를 안보여줘도 됩니다.
- 검색 결과에 없는 정보는 추측하지 않습니다. (예시: 회색 상품이 있습니다.)
- 이전 질문과 연관지어서 답변하지마세요.
- 상품의 총 개수 같은 질문은 모른다고 하세요
- 상품이나 상품추천 질문아니면 답변하지말고 상품관련 질문만 해달라고 하세요
- IT 기술 질문 무시 예) 파이썬, 랭체인 등등
- 유사도 0.8 이상의 같은 종류상품이 아니면 같은상품 아니라고 말하고 최대한 비슷한 상품을 추천했다고 말합니다.
"""


def extract_message_text(content: Any) -> str:
    """
    AIMessage.content에서 사용자에게 보여줄 실제 텍스트만 추출합니다.

    LLM 제공자나 라이브러리 버전에 따라 AIMessage.content의 형식이
    다를 수 있기 때문에 이를 하나의 문자열 형식으로 통일합니다.

    지원하는 입력 형식:
    1. 일반 문자열
       예: "안녕하세요!"

    2. Gemini 등의 content block 리스트
       예:
       [
           {
               "type": "text",
               "text": "안녕하세요!",
               "extras": {"signature": "..."}
           }
       ]

    3. 문자열과 딕셔너리가 혼합된 리스트
       예:
       [
           "첫 번째 문장",
           {"type": "text", "text": "두 번째 문장"}
       ]

    반환값:
    - 추출된 모든 텍스트를 줄바꿈으로 연결한 문자열
    - 추출할 수 있는 텍스트가 없으면 빈 문자열
    """

    # content가 이미 문자열이면 별도의 변환이 필요 없습니다.
    #
    # strip()을 사용하여 문자열 앞뒤에 붙어 있을 수 있는
    # 공백, 줄바꿈, 탭 문자를 제거한 뒤 반환합니다.
    #
    # 예:
    # "  안녕하세요!\n" → "안녕하세요!"
    if isinstance(content, str):
        return content.strip()

    # Gemini를 포함한 일부 모델은 응답을 단순 문자열이 아니라
    # 여러 개의 content block이 들어 있는 리스트로 반환할 수 있습니다.
    #
    # 예:
    # [
    #     {
    #         "type": "text",
    #         "text": "안녕하세요!",
    #         "extras": {"signature": "..."}
    #     }
    # ]
    #
    # 따라서 content가 리스트라면 각 항목을 순회하면서
    # 실제 텍스트에 해당하는 값만 수집합니다.
    if isinstance(content, list):
        # content block에서 추출한 텍스트를 순서대로 저장할 리스트입니다.
        #
        # 하나의 응답이 여러 텍스트 블록으로 나뉘어 있을 수 있으므로,
        # 즉시 반환하지 않고 모든 텍스트를 모은 후 마지막에 합칩니다.
        texts = []

        # content 리스트의 각 블록을 처음부터 순서대로 확인합니다.
        for block in content:
            # 리스트 안에 문자열이 직접 들어 있는 경우입니다.
            #
            # 예:
            # [
            #     "첫 번째 문장",
            #     "두 번째 문장"
            # ]
            #
            # 문자열을 그대로 결과 목록에 추가합니다.
            if isinstance(block, str):
                texts.append(block)

            # 블록이 딕셔너리인 경우에는 "text" 필드를 확인합니다.
            #
            # Gemini 응답은 일반적으로 다음과 같은 구조를 가질 수 있습니다.
            #
            # {
            #     "type": "text",
            #     "text": "실제 사용자에게 보여줄 답변",
            #     "extras": {
            #         "signature": "모델 내부 메타데이터"
            #     }
            # }
            elif isinstance(block, dict):
                # 딕셔너리에서 실제 답변이 들어 있는 "text" 값을 가져옵니다.
                #
                # "text" 키가 존재하지 않으면 get()은 None을 반환하므로
                # KeyError가 발생하지 않습니다.
                text = block.get("text")

                # text 값이 실제 문자열인 경우에만 결과에 추가합니다.
                #
                # None, 리스트, 딕셔너리 같은 예상하지 못한 형식은
                # 사용자 답변에 포함하지 않고 안전하게 무시합니다.
                if isinstance(text, str):
                    texts.append(text)

        # 수집된 여러 텍스트 블록을 줄바꿈 문자로 연결합니다.
        #
        # 예:
        # ["첫 번째 문장", "두 번째 문장"]
        #
        # 결과:
        # "첫 번째 문장\n두 번째 문장"
        #
        # 마지막 strip()은 전체 결과 앞뒤의 불필요한 공백과
        # 줄바꿈을 제거합니다.
        return "\n".join(texts).strip()

    # content가 문자열이나 리스트가 아닌 예상하지 못한 형식이면
    # 오류를 발생시키지 않고 빈 문자열을 반환합니다.
    #
    # 예:
    # None, 숫자, 임의 객체, 지원하지 않는 응답 형식
    return ""


def chatbot(state: AgentState):
    """
    검색(QDRANT SEARCH) 도구를 바인딩 한 LLM 모델에 현재 메시지 상태를 입력하여 응답을 생성합니다.
    질문이 주어지면 검색 도구를 도구호출 하거나 일반 답변하며 종료할지 결정할 수 있습니다.
    """
    label_text = (state.get("label_text") or "").strip()
    has_image = (
        state.get("is_image_collection", False)
        and state.get("query_vector") is not None
    )
    print("----- [CHATBOT] -----")
    # system_prompt를 MessagesState에 추가하기 위해 AI Message로 변환
    # system_message = AIMessage(content=system_prompt)
    system_message = SystemMessage(content=system_prompt)
    # state에 system 메시지를 먼저 추가하고 나머지 메시지들을 뒤에 이어 붙입니다.
    messages = [system_message]

    # 라벨 분석 성공 여부와 관계없이 이미지 첨부 사실을 명시
    if has_image:
        image_context = (
            "[첨부 이미지 분석 정보]\n"
            "- 상품 이미지가 이미 첨부됨\n"
            "- 이미지를 다시 첨부하거나 업로드해 달라고 요청하지 마세요.\n"
        )
        if label_text:
            image_context += (
                f"- 상품 종류: {label_text}\n"
                "- 위 상품 종류는 이미지에서 감지한 참고 정보입니다.\n"
            )
        image_context += (
            "- 사용자가 지정한 색상이나 조건이 있으면 "
            "이미지의 기존 속성보다 사용자 조건을 우선하세요."
        )
        image_context_message = HumanMessage(content=image_context)
        messages.append(image_context_message)

    messages.extend(state["messages"])

    print("----- [LLM INPUT] -----")
    for message in messages:
        print(type(message).__name__, repr(message.content))

    response = llm_with_tools.invoke(messages)
    answer_text = suppress_redundant_image_request(
        extract_message_text(response.content),
        has_image=has_image,
    )

    print(f"label_text: , {label_text}")
    print("원본 content:", repr(response.content))
    print("추출된 답변:", repr(answer_text))
    print("response tool_calls:", response.tool_calls)

    result = {
        # LLM 응답은 항상 대화 기록에 추가
        "messages": [response],
    }

    if not response.tool_calls:
        # 도구 호출이 없으면 이 응답이 최종 답변이므로 저장
        result["answer"] = answer_text

    # LangGraph state에 적용할 변경사항 반환
    return result


def route_tools(state: AgentState):
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    has_image = (
        state.get("is_image_collection", False)
        and state.get("query_vector") is not None
    )

    print("----- [ROUTE DEBUG] -----")
    print("is_image_collection:", state.get("is_image_collection"))
    print("has_query_vector:", state.get("query_vector") is not None)
    print("has_image:", has_image)
    print("content:", repr(last_message.content))
    print("tool_calls:", tool_calls)

    # 이미지가 첨부된 요청은 상품 검색 우선
    if has_image:
        print("----- [ROUTE IMAGE TO QDRANT SEARCH] -----")
        return "tools"

    # 이미지 요청이 아니고 도구 호출도 없으면 종료
    if not tool_calls:
        print("----- [ROUTE END: NO TOOL CALL] -----")
        return END

    # 어떤 도구인지 확인
    tool_name = last_message.tool_calls[0]["name"]

    if tool_name == "products_images_search":
        print("----- [ROUTE TOOLS QDRANT SEARCH] -----")
        return "tools"  # →  QDRANT SEARCH
    elif tool_name == "handle_small_talk":
        print("----- [ROUTE small_talk] -----")
        # 핵심 수정: 그래프 이미지에 있는 노드 이름과 똑같이 맞춰줍니다!
        return "small_talk"
    else:
        print("----- [END] -----")
        return END


def qdrant_search(state: AgentState):
    """
    현재 질문 또는 멀티모달 벡터를 기반으로 상품 정보(문서)를 검색합니다.
    """
    print("----- [QDRANT + SQLLITE SEARCH] -----")

    query_vector = state.get("query_vector")
    is_image_collection = state.get("is_image_collection")
    print("is_image_collection", is_image_collection)
    if is_image_collection:
        collection_name = COLLECTION_NAME_IMG
        print(f"----- [QDRANT SEARCH] : IMAGE COLLECTION ----- {collection_name}")
    else:
        collection_name = COLLECTION_NAME_DESC
        print(f"----- [QDRANT SEARCH] : TEXT COLLECTION ----- {collection_name}")

    structured_results = []
    # [1] 벡터 검색
    if query_vector:
        results = client.query_points(
            collection_name=COLLECTION_NAME_IMG,
            query=query_vector,
            limit=TOP_K,
            with_payload=True,
            with_vectors=False,
        ).points

        seen_ids = set()
        context = "검색된 상품 목록은 다음과 같습니다:\n"

        for idx, point in enumerate(results, 1):
            payload = point.payload or {}

            product_id = payload.get("product_id")
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

            structured_results.append(
                {
                    "score": round(score, 4),
                    "product_id": product_id,
                    "description": description,
                    "image_url": image_url,
                }
            )

            context += f"[{product_id}번 상품] (유사도: {round(score, 4)})\n"
            context += f"- 상품 설명: {description}\n"

    if structured_results:
        # print("qdrant조회 결과 O")
        db_results = qdrant_to_sql(structured_results)
        if db_results:
            structured_results = db_results
            print(f"- DB조회 결과: {structured_results}")
            context += f"- DB조회 결과: {structured_results}\n"
        else:
            print("DB조회 실패")

    # Tool 호출에 대한 응답 메시지(검색 결과) 생성
    last_message = state["messages"][-1]

    # 리턴할 때 search_results 필드에 우리가 만든 JSON 리스트를 같이 담아서 넘김
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        tool_call_id = last_message.tool_calls[0]["id"]
        tool_message = ToolMessage(
            content=context, name="products_images_search", tool_call_id=tool_call_id
        )
        return {
            "messages": [tool_message],
            "context": context,
            "search_results": structured_results,
            "answer": context,
        }
    else:
        return {
            "messages": [AIMessage(content=context)],
            "context": context,
            "search_results": structured_results,
            "answer": context,
        }


def small_talk(state: AgentState):
    """
    일상 대화 도구가 호출되었을 때 실행되는 노드입니다.
    """
    print("----- [HANDLE SMALL TALK NODE] -----")

    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        tool_call = last_message.tool_calls[0]
        tool_call_id = tool_call["id"]

        # LLM이 도구를 호출하면서 작성한 'response' 파라미터 값을 꺼냅니다.
        args = tool_call.get("args", {})

        # 만약 LLM이 파라미터를 깜빡하고 안 넘겼을 경우를 대비해 기본값을 설정해 줍니다.
        content = args.get(
            "response", "안녕하세요! 쇼핑 어시스턴트입니다. 무엇을 도와드릴까요?"
        )

        tool_message = ToolMessage(
            content=content, name="handle_small_talk", tool_call_id=tool_call_id
        )

        # 검색 결과가 없으므로 search_results는 빈 리스트([])로 넘겨 프론트엔드 에러를 방지합니다.
        return {
            "messages": [tool_message],
            "context": content,
            # 2.  프론트엔드가 화면에 띄울 수 있게 여기서 "answer"에 직접 대답을 넣어줍니다!
            "answer": content,
            # 3. 엑스박스 방지를 위해 빈 배열을 줍니다.
            "search_results": [],
        }
    else:
        # 혹시 도구 호출 정보가 없는 예외 상황을 위한 안전장치
        return {
            "messages": [
                AIMessage(content="무엇을 도와드릴까요? 원하시는 상품을 말씀해주세요.")
            ],
            "context": "",
            "search_results": [],
        }


### 해결: LLM 의존 제거, 코드 기반 필터링으로 변경


def context_organizer(state: AgentState):
    search_results = state.get("search_results") or []
    label_text = state.get("label_text") or ""

    # label_text가 리스트로 들어오는 경우 문자열로 정규화
    if isinstance(label_text, list):
        label_text = ", ".join(str(label) for label in label_text)

    # 가능하면 messages를 역순 탐색하기보다
    # state에 저장된 원본 질문을 우선 사용
    user_message = state.get("original_question") or state.get("question") or ""

    # 질문이 state에 없을 때만 HumanMessage에서 찾음
    if not user_message:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                user_message = extract_message_text(message.content)
                break

    # 검색 결과 자체가 없으면 LLM 필터링을 호출하지 않음
    if not search_results:
        return {
            "context": "관련 있는 상품을 찾지 못했습니다.",
            "search_results": [],
        }

    # LLM에는 필요한 필드만 전달하여 토큰 사용량 감소
    simplified_results = []

    for item in search_results:
        item_id = item.get("id")

        # ID가 없으면 LLM이 선택할 수 없으므로 제외
        if item_id is None:
            continue

        # 카테고리 정보를 공백으로 합치고 빈 값은 제거
        categories = " ".join(
            str(category).strip()
            for category in [
                item.get("category1"),
                item.get("category2"),
                item.get("category3"),
                item.get("category4"),
            ]
            if category
        )

        simplified_results.append(
            {
                "id": str(item_id),
                "title": str(item.get("title") or ""),
                "category": categories,
            }
        )

    filter_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            당신은 벡터 검색 결과에서 명백히 잘못된 상품만 제거하는 필터입니다.

            제거해야 하는 상품의 ID만 반환하세요.

            판단 규칙:
            - 벡터 검색 결과는 기본적으로 유지하세요.
            - 사용자 요청 또는 이미지 라벨과 상품 종류가 명백히 다른 경우만 제거하세요.
            - 판단하기 어렵거나 정보가 부족하면 반드시 유지하세요.
            - 상품 제목이나 설명에 색상, 소재, 크기가 없다는 이유로 제거하지 마세요.
            - 사용자 요청이 색상 조건만 포함하면 이미지 라벨로 상품 종류를 판단하세요.
            - 이미지 라벨과 상품 제목의 종류가 일치하면 유지하세요.
            - `원피스`, `dress`, `one-piece garment`, `day dress`는 같은 종류입니다.
            - `야구 모자`, `baseball cap`, `cricket cap`은 같은 종류의 상품으로 판단합니다.
            - 반드시 제공된 상품 ID 중에서만 선택하세요.
            - 제거할 상품이 없으면 정확히 `없음`만 반환하세요.
            - 제거할 상품이 있으면 제거할 ID만 쉼표로 구분하세요.
            - JSON, 설명, 문장, 마크다운은 출력하지 마세요.

            예시:

            사용자 요청: 빨간색으로
            이미지 라벨: one-piece garment, day dress, dress
            상품 목록:
            - ID 731, 레이어드 원피스
            - ID 900, 크로스백

            출력:
            900
                        """.strip(),
            ),
            (
                "user",
                """
            사용자 요청: {user_message}
            이미지 라벨: {label_text}
            검색된 상품 목록: {search_results}
                        """.strip(),
            ),
        ]
    )

    filter_chain = filter_prompt | llm

    print(f"\n----- User message: {user_message} -----")
    print(f"----- Image labels: {label_text} -----")
    print(f"----- Simplified results: {simplified_results} -----")

    filter_response = filter_chain.invoke(
        {
            "user_message": user_message,
            "label_text": label_text,
            # Python repr보다 JSON이 모델이 읽기 쉬움
            "search_results": json.dumps(
                simplified_results,
                ensure_ascii=False,
            ),
        }
    )

    # Gemini는 content block 리스트를 반환할 수 있으므로
    # 직접 .strip()을 호출하지 않고 텍스트 추출 함수 사용
    response_text = extract_message_text(filter_response.content)

    print(f"- LLM 필터 원본 응답: {filter_response.content!r}")
    print(f"- LLM 필터 텍스트: {response_text!r}")

    allowed_ids = {
        str(item["id"]) for item in search_results if item.get("id") is not None
    }

    response_text = extract_message_text(filter_response.content)

    parse_failed = False

    if response_text.strip() == "없음":
        # 제거할 상품이 없음 → 전체 유지
        removed_ids = set()
    else:
        extracted_ids = set(re.findall(r"\b\d+\b", response_text))

        if not extracted_ids:
            # 응답을 해석할 수 없으면 안전하게 전체 유지
            parse_failed = True
            removed_ids = set()
        else:
            # 실제 검색 결과에 존재하는 ID만 제거 대상으로 인정
            removed_ids = extracted_ids & allowed_ids

    # 명백히 제거 대상으로 판정된 상품만 제외
    effective_results = [
        item
        for item in search_results
        if (item.get("id") is not None and str(item["id"]) not in removed_ids)
    ]

    kept_ids = allowed_ids - removed_ids

    print(
        f"- 필터링 전: {len(search_results)}건 → 필터링 후: {len(effective_results)}건"
    )
    print(f"- 유지된 ID: {kept_ids}")
    print(f"- 제거된 ID: {removed_ids}")

    if parse_failed:
        print("- 필터 응답 파싱 실패: 모든 검색 결과를 유지합니다.")

    effective_ids = {
        str(item["id"]) for item in effective_results if item.get("id") is not None
    }

    removed_ids = allowed_ids - effective_ids

    print(
        f"- 필터링 전: {len(search_results)}건 → 필터링 후: {len(effective_results)}건"
    )
    print(f"- 유지된 ID: {effective_ids}")

    if removed_ids:
        print(f"- 제거된 ID: {removed_ids}")

    if effective_results:
        filtered_context = "검색된 상품 목록은 다음과 같습니다:\n" + json.dumps(
            effective_results,
            ensure_ascii=False,
        )
    else:
        filtered_context = "사용자의 요청과 관련 있는 상품을 찾지 못했습니다."

    return {
        "context": filtered_context,
        "search_results": effective_results,
    }


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
    return {
        "question": better_question.content,
        "messages": [better_question],
        "retry_num": state["retry_num"] + 1 if state.get("retry_num") else 1,
    }


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
    return {
        "question": better_question.content,
        "messages": [better_question],
        "retry_num": state["retry_num"] + 1 if state.get("retry_num") else 1,
    }


def generate(state: AgentState):
    """
    검색된 문서 및 질문을 기반으로 답변을 생성합니다.
    """
    print("----- [GENERATE] -----")
    question = state.get("question", "")
    context = state.get("context", "")
    retry_num = state.get("retry_num", 0)
    has_image = bool(state.get("is_image_collection", False))
    image_status = "첨부됨" if has_image else "첨부되지 않음"

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
                    "대신 이런 상품은 어떠신가요?" 라며 대안을 제안하는 가이드를 작성하세요.
                    이미지 첨부 여부가 `첨부됨`이면 이미지를 다시 첨부하거나 업로드해 달라고 요청하지 마세요.""",
                ),
                (
                    "user",
                    "이미지 첨부 여부: {image_status} \n\n질문: {question} \n\n검색 결과: {context} \n\n답변:",
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
                    
                    [절대 지켜야 할 답변 규칙]  !!!중요!!!
                    1. 환각 금지: 절대로 가상의 상품명, 가격, 색상을 지어내지 마세요.
                    2. 반말금지: 항상 존댓말을 사용하세요.
                    3. 목록 나열 금지: 검색 결과(context)에 있는 상품 정보를 줄글이나 번호 매기기(1, 2, 3...)로 길게 나열하지 마세요. (사용자 화면 하단에 상품 카드가 자동으로 따로 표시됩니다.)
                    4. 추천 하는 이유를 검색 결과(context)에 맞추어 간략하게 설명해주세요.
                    5. 불필요한 사족(예: '제가 제공한 검색 결과 중에는~')은 모두 빼고 자연스럽게 대화하듯 말하세요.
                    6. "유사도" 라는말 금지 !!
                    7. 상품 설명에 질문한 상품이 없으면 찾는 상품 없다고 말하세요
                    8. 링크는 알려주지마세요
                    9. 이미지 첨부 여부가 `첨부되지 않음`이고 찾는 상품이 없을 때만 상품 이미지 첨부를 제안하세요.
                       이미지 첨부 여부가 `첨부됨`이면 이미지 첨부나 업로드를 다시 요청하지 마세요.
                    10. question에 맞춰서 대답 하세요
                    """,
                ),
                (
                    "user",
                    "이미지 첨부 여부: {image_status} \n\n질문: {question} \n\n검색 결과: {context} \n\n안내 멘트:",
                ),
            ]
        )

    rag_chain = rag_prompt | llm
    response = rag_chain.invoke(
        {
            "question": question,
            "context": context,
            "image_status": image_status,
        }
    )
    answer = suppress_redundant_image_request(
        extract_message_text(response.content),
        has_image=has_image,
    )
    return {
        "question": question,
        "answer": answer,
        "messages": [AIMessage(content=answer)],
        "retry_num": state["retry_num"] + 1 if state.get("retry_num") else 1,
    }

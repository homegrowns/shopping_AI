import os
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field
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


def generate_shopping_answer(
    question: str,
    candidates: list[dict],
    image_labels: list[str] | None = None,
) -> str:
    """
    이미지/텍스트 유사도 검색(CLIP + Qdrant)으로 찾은 후보 상품 목록을 받아,
    사용자의 질문 의도에 맞는 카테고리의 상품만 추려서 자연어 추천 답변을 생성한다.

    후보 목록은 이미 스코어(유사도) 기준으로만 필터링된 상태라, 카테고리가
    맞지 않는 상품(예: 신발을 찾는데 액세서리가 섞이는 경우)이 섞여 있을 수 있다.
    상품 데이터에 별도 카테고리 컬럼이 없기 때문에, 카테고리 판단 자체는
    코드가 아니라 이 프롬프트를 통해 LLM이 수행한다.

    image_labels가 주어지면(Vision Label Detection 결과, 보통 영어) 참고
    컨텍스트로 프롬프트에 함께 넣는다. 사용자 텍스트와 문자열이 정확히
    겹치는지는 확인하지 않고, LLM이 의미/맥락으로 판단하도록 맡긴다.
    (예: 사용자가 "이런 비슷한 상품 추천해줘"처럼 막연하게 질문해도,
    라벨이 "video projector, gadget"이면 그 의미를 참고해 카테고리를 유추)

    Args:
        question: 사용자가 입력한 텍스트 질문
        candidates: [{"product_id":..., "title":..., "score":...}, ...] 형태의 후보 목록
        image_labels: 첨부 이미지에서 감지된 라벨 목록 (없으면 None)

    Returns:
        LLM이 생성한 자연어 추천 답변 (str)
    """
    print("----- [GENERATE SHOPPING ANSWER] -----")

    if not candidates:
        return "조건에 맞는 유사 상품을 찾지 못했어요."

    candidates_text = "\n".join(
        f"- (id: {c.get('product_id')}) {c.get('title') or '제목 없음'}"
        for c in candidates
    )

    if image_labels:
        image_context = (
            f"\n\n첨부된 이미지에서 감지된 대략적인 내용(참고용, 영어일 수 있음): "
            f"{', '.join(image_labels)}"
        )
    else:
        image_context = ""

    shopping_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 쇼핑몰 AI 어시스턴트입니다.
                아래 "후보 상품 목록"은 사용자가 업로드한 이미지 또는 입력한 텍스트와
                시각적/의미적으로 유사도가 높다고 판단되어 검색된 상품들입니다.
                다만 이 목록은 유사도 점수만으로 걸러진 것이라, 사용자의 질문 의도와
                맞지 않는 카테고리의 상품이 섞여 있을 수 있습니다
                (예: 신발을 찾는 질문인데 액세서리나 가방이 섞여 있는 경우).

                이미지가 첨부된 경우, 이미지에서 감지된 내용(라벨)이 함께
                주어질 수 있습니다. 사용자의 질문이 "이런 거 추천해줘"처럼
                막연하더라도, 이 라벨의 "의미"를 참고해서(문자열이 정확히
                일치하지 않아도 됩니다. 예: 라벨 "video projector"는
                "로봇청소기"나 "빔프로젝터"와 의미적으로 연결될 수 있습니다)
                실제로 같은 카테고리인지 스스로 판단하세요.

                [중요 규칙]
                1. 사용자의 질문 의도(및 이미지 라벨의 의미)에 맞는 카테고리의
                   상품만 골라서 언급하세요.
                2. 후보 목록에 없는 상품을 절대 지어내지 마세요.
                3. 의도에 맞는 상품이 하나도 없다면, 그렇다고 솔직하게 답변하세요.
                4. 답변은 친절하고 간결한 한국어 문장으로 작성하세요
                   (번호로 나열하기보다는 자연스러운 추천 대화체로).
                """,
            ),
            (
                "user",
                "사용자 질문: {question}{image_context}\n\n후보 상품 목록:\n{candidates_text}\n\n답변:",
            ),
        ]
    )

    chain = shopping_prompt | llm
    response = chain.invoke(
        {
            "question": question,
            "image_context": image_context,
            "candidates_text": candidates_text,
        }
    )
    return response.content


class QueryDescriptiveness(BaseModel):
    """사용자 텍스트가 검색 쿼리로 쓸 만큼 구체적인 묘사인지 판단"""

    is_descriptive: bool = Field(
        description="사용자 텍스트가 색상/종류/브랜드/용도 등 검색에 실제로 "
        "도움이 되는 구체적 단서를 포함하면 True. "
        "'이런 거 찾아줘', '비슷한 상품 추천해줘'처럼 첨부 이미지를 "
        "가리키기만 하는 막연한 요청이면 False."
    )
    reason: str = Field(description="판단 이유를 한 문장으로")


def is_descriptive_query(text: str) -> bool:
    """
    이미지+텍스트를 함께 입력했을 때, 이 텍스트를 검색 쿼리 벡터에
    포함시킬지 판단하기 위한 함수.

    "이런 거 찾아줘"처럼 막연한 텍스트를 CLIP으로 임베딩하면 실질적인
    의미 정보가 거의 없는 벡터가 나오는데, 이걸 image_vector와 동일
    비중으로 평균 내면 오히려 image_vector가 갖고 있던 정확한 위치
    정보를 흐려서 검색 정밀도가 떨어질 수 있다.

    그래서 텍스트가 "구체적 묘사"인지 LLM으로 판단해서:
    - True(구체적)면: 검색 쿼리에 텍스트도 포함 (기존 50:50 방식)
    - False(막연함)면: 검색 쿼리에서는 텍스트를 빼고 이미지만 사용
      (단, 텍스트 자체는 여전히 generate_shopping_answer()의 질문으로는
      그대로 전달되어 답변 생성에는 영향을 준다)

    판단 자체가 실패하면(LLM 에러 등) 안전하게 True로 간주해
    기존 동작(텍스트도 검색에 반영)을 유지한다.
    """
    print("----- [CHECK QUERY DESCRIPTIVENESS] -----")

    classify_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 사용자의 쇼핑 검색 텍스트가 "구체적인 묘사"인지
                "막연한 요청"인지 판단하는 분류기입니다.

                - 구체적 묘사: 색상, 종류, 브랜드, 용도, 특징, 가격대 등
                  검색에 실제로 도움이 되는 단서를 포함
                  (예: "초록색 구두", "가벼운 로봇청소기", "20만원 이하 무선 이어폰")
                - 막연한 요청: 첨부한 이미지를 가리키기만 하고 구체적인
                  단서가 전혀 없는 일반적인 문구
                  (예: "이런 거 찾아줘", "비슷한 상품 추천해줘", "이거랑 똑같은 거 있어?")
                """,
            ),
            ("user", "사용자 텍스트: {text}"),
        ]
    )

    structured_llm = llm.with_structured_output(QueryDescriptiveness)
    chain = classify_prompt | structured_llm

    try:
        result = chain.invoke({"text": text})
        print(
            f"[Query Descriptiveness] is_descriptive={result.is_descriptive} "
            f"({result.reason})"
        )
        return result.is_descriptive
    except Exception as e:
        print(f"[Query Descriptiveness] 판단 실패, 안전하게 구체적으로 간주: {e}")
        return True


class RelevantProducts(BaseModel):
    """이미지 라벨과 의미적으로 연관된 상품 id 목록"""

    relevant_product_ids: list[str] = Field(
        description="후보 상품 중, 이미지 라벨과 같은 카테고리/의미로 판단되는 상품의 id 목록. "
        "관련 있는 상품이 하나도 없으면 빈 리스트."
    )


def filter_products_by_labels(
    image_labels: list[str],
    candidates: list[dict],
) -> list[str] | None:
    """
    텍스트 질문 없이 이미지만 업로드된 경우에 사용.
    이미지에서 감지된 라벨(영어)과 후보 상품 목록(제목은 한국어일 수 있음)을
    LLM에게 함께 주고, "문자열이 정확히 일치하지 않아도" 의미적으로 같은
    카테고리라고 판단되는 상품의 id만 구조화된 출력으로 받아온다.

    반환값:
    - list[str]: LLM이 실제로 판단을 마친 결과. 빈 리스트([])면
      "연관된 상품이 정말 하나도 없다"는 뜻이므로 호출부는 그대로 빈 결과를 써야 한다.
    - None: 라벨이 없거나 LLM 호출 자체가 실패해 "판단을 못한" 경우.
      이때만 호출부가 원본 후보 목록으로 폴백해야 한다.
    """
    if not candidates or not image_labels:
        print("[Filter By Labels] 라벨 또는 후보가 없어 판단 불가")
        return None

    candidates_text = "\n".join(
        f"- (id: {c.get('product_id')}) {c.get('title') or '제목 없음'}"
        for c in candidates
    )

    filter_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 쇼핑몰 이미지 검색 결과를 정리하는 어시스턴트입니다.
                사용자는 텍스트 질문 없이 이미지 하나만 업로드했습니다.
                아래 "이미지 라벨"은 Vision API가 그 이미지에서 감지한 대략적인
                내용입니다 (영어일 수 있습니다).
                "후보 상품 목록"은 그 이미지와 시각적 유사도가 높아 검색된
                상품들이지만, 유사도 점수만 본 것이라 실제로는 관련 없는
                카테고리의 상품이 섞여 있을 수 있습니다.

                라벨과 상품명이 문자 그대로 일치하지 않아도 괜찮습니다.
                의미/맥락으로 판단하세요
                (예: 라벨 "video projector, gadget"이면 로봇청소기, 빔프로젝터,
                전자기기류 상품과 의미적으로 연관될 수 있습니다).

                이미지 라벨과 같은 카테고리/의미로 보이는 상품의 id만
                relevant_product_ids에 담아 반환하세요. 후보에 없는 id를
                지어내지 마세요. 관련 있는 상품이 하나도 없다면 빈 리스트를
                반환하세요.
                """,
            ),
            (
                "user",
                "이미지 라벨: {image_labels}\n\n후보 상품 목록:\n{candidates_text}",
            ),
        ]
    )

    structured_llm = llm.with_structured_output(RelevantProducts)
    chain = filter_prompt | structured_llm

    try:
        result = chain.invoke(
            {
                "image_labels": ", ".join(image_labels),
                "candidates_text": candidates_text,
            }
        )
        print(f"[Filter By Labels] 연관 상품으로 판단된 id: {result.relevant_product_ids}")
        return result.relevant_product_ids
    except Exception as e:
        print(f"[Filter By Labels] 구조화 출력 실패, 판단 불가로 처리: {e}")
        return None
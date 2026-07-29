# 환경 변수 설정 (.env)
# ==========================================
# 아래 내용을 복사하여 .env 파일을 만들고 실제 값을 입력하세요.
# (주의: .env 파일은 절대 GitHub에 커밋하지 마세요!)
# ==========================================

```python
# 1. API Keys (필수)
GOOGLE_API_KEY="본인의_Google_API_Key_입력"
TAVILY_API_KEY="본인의_Tavily_API_Key_입력"
LANGSMITH_API_KEY="본인의_LangSmith_API_Key_입력" # (옵션: 로깅용)
# 2. Google Cloud 자격 증명 파일 경로
GOOGLE_APPLICATION_CREDENTIALS="/절대경로를_입력하세요/google-credentials.json"
# 3. Database 설정 (SQLite 및 Qdrant)
SQL="sqlite:////절대경로를_입력하세요/products.db"
QDRANT_HOST="서버_IP_주소_입력"
QDRANT_PORT=6333
QDRANT_COLLECTION="products_images" # 사용하는 컬렉션 이름
# 4. 애플리케이션 및 검색 튜닝 옵션
TOP_K=8         # 가져올 최대 검색 결과 수
SCORE=0.3       # 결과로 인정할 최소 유사도 점수 (Threshold)
ENV="dev"       # 실행 환경 (dev / prod)
```

```
if os.getenv("ENV") == "prod":
    from langchain_aws import ChatBedrock
    llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet...")
    print("(edges.py) LLM: ", "prod")

elif os.getenv("ENV") == "local":
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    print("(edges.py) LLM: ", "local")

elif os.getenv("ENV") == "dev":
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.1")
    print("(edges.py)LLM: ", "dev")
```
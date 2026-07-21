### .env 생성

```python
GOOGLE_API_KEY="키 생성 필요"

TAVILY_API_KEY="키 생성 필요"

LANGSMITH_API_KEY="키 생성 필요"

File_Path="경로/한글맞춤법_표준어규정_해설.pdf"

DB_PATH="경로/chroma_db"

SQL="sqlite:///경로/products.db"

QDRANT_HOST="서버 ip"

QDRANT_PORT="서버 포트번호"

QDRANT_COLLECTION="서버 컬렉션 이름"

ENV="dev"
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
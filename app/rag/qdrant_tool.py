from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import create_retriever_tool
from langchain_core.tools import tool

from app.image_embedding_similarity.embedder import ClipEmbedder
from app.image_embedding_similarity.qdrant_utils import get_client, COLLECTION_NAME

# 1. LangChain 호환용 임베딩 래퍼(Adapter) 생성
class LangChainClipEmbedder(Embeddings):
    def __init__(self):
        # 원본 코드의 싱글톤 인스턴스 가져오기
        self.clip = ClipEmbedder.get_instance()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 여러 텍스트를 임베딩할 때 (문서 추가 시 주로 사용)
        # SentenceTransformer는 리스트 입력을 받아 배치 처리가 가능하므로 직접 encode 호출이 효율적입니다.
        vectors = self.clip.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        # 에이전트(LLM)가 검색 툴을 사용할 때(질문할 때) 호출됨
        return self.clip.embed_text(text)

    def embed_text(self, text: str) -> list[float]:
        """CLIP의 텍스트 인코더로 텍스트를 이미지와 같은 공간에 임베딩."""
        vector = self.clip.model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )
        return vector.tolist()


# 2. Qdrant 클라이언트 및 임베더 준비
client = get_client()
langchain_embedder = LangChainClipEmbedder()

# 3. LangChain 벡터스토어 초기화 (기존 컬렉션 안전 연결)
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=langchain_embedder,
)

# 4. 리트리버 및 도구 생성
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# --- [ 1. 밖에서 세팅하는 곳 (도구 준비) ] ---
products_images_search_tool = create_retriever_tool(
    retriever,
    name="products_images_search",
    description=(
        "Search for similar products in the product image collection. "
        "[IMPORTANT] If the user's input is in Korean, you MUST translate it into English before searching. "
        "(e.g., '검은 나이키 운동화' -> 'black nike sneakers')"
    )
)
"""
CLIP 기반 이미지/텍스트 임베딩 모듈.
sentence-transformers의 clip-ViT-B-32 사용 (출력 차원 512).
CLIP은 이미지와 텍스트를 같은 임베딩 공간에 매핑하므로,
이미지 검색뿐 아니라 텍스트로도, 이미지+텍스트 조합으로도 검색이 가능하다.
"""
from PIL import Image
from sentence_transformers import SentenceTransformer

MODEL_NAME = "clip-ViT-B-32"
VECTOR_SIZE = 512


class ClipEmbedder:
    _instance = None

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    @classmethod
    def get_instance(cls) -> "ClipEmbedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed_image(self, image: Image.Image) -> list[float]:
        image = image.convert("RGB")
        vector = self.model.encode(
            image, convert_to_numpy=True, normalize_embeddings=True
        )
        return vector.tolist()

    def embed_image_path(self, path: str) -> list[float]:
        image = Image.open(path)
        return self.embed_image(image)

    def embed_text(self, text: str) -> list[float]:
        """CLIP의 텍스트 인코더로 텍스트를 이미지와 같은 공간에 임베딩."""
        vector = self.model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )
        return vector.tolist()
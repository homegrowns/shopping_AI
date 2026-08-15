import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import tool

def handle_small_talk(response: str) -> str:
    """
    사용자가 상품 검색이 아닌 일상적인 인사나 잡담(Small talk)혹은 의미없는 단어를 봤을때 이 도구를 호출하세요.
    response 파라미터에는 사용자에게 전달할 다정하고 친절한 인사말이나 답변을 직접 작성해서 넣어주세요.
    """
    return response
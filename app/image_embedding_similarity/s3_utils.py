"""
사용자가 검색을 위해 업로드한 이미지를 S3에 저장한다.
경로 규칙: s3://ecommerce-tool-calling-agent/input/{session_id}/{n}.jpg
- session_id: 브라우저 세션(탭)마다 하나씩, 클라이언트가 sessionStorage로 관리해 전달
- n: 해당 세션 내에서 1부터 시작하는 순번 (서버가 메모리에서 카운트)

주의: 순번 카운터는 프로세스 메모리에 저장되므로, --reload로 서버가
재시작되면 카운터가 초기화된다. 여러 워커/여러 서버로 확장할 경우
Redis 등 외부 저장소로 옮기는 것을 권장한다 (지금 규모에서는 불필요).
"""
import io
import os
import threading

import boto3
from PIL import Image

S3_BUCKET = "ecommerce-tool-calling-agent"
S3_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")

_s3_client = boto3.client("s3", region_name=S3_REGION)
_session_counters: dict[str, int] = {}
_lock = threading.Lock()


def _next_index_for_session(session_id: str) -> int:
    with _lock:
        current = _session_counters.get(session_id, 0) + 1
        _session_counters[session_id] = current
        return current


def upload_input_image(session_id: str, image: Image.Image) -> str:
    """
    PIL 이미지를 JPEG로 인코딩해 S3에 업로드하고, 저장된 S3 key를 반환한다.
    """
    index = _next_index_for_session(session_id)
    key = f"input/{session_id}/{index}.jpg"

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    buf.seek(0)

    _s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=buf.getvalue(),
        ContentType="image/jpeg",
    )
    return key
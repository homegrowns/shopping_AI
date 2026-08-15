# ============================================================
# Stage 1: Build
# ============================================================
FROM python:3.12-slim AS builder

WORKDIR /shopping_assistant

# uv 대신 파이썬 내장 venv를 사용하여 가상환경 생성
RUN python -m venv /shopping_assistant/.venv
ENV PATH="/shopping_assistant/.venv/bin:$PATH"

# pip 최신화
RUN pip install --upgrade pip --no-cache-dir

# [핵심] 가장 먼저 CPU 버전 PyTorch를 쐐기 박듯이 강제 설치합니다.
# 이렇게 선점해두면 이후에 어떤 패키지를 깔아도 무거운 CUDA 버전이 설치되지 않습니다.
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# 설정 파일(requirements.txt) 복사
COPY requirements.txt ./

# requirements.txt 에 명시된 나머지 패키지들을 설치
# (이미 CPU 버전 torch가 있으므로 GPU 버전을 새로 다운받지 않습니다)
RUN pip install -r requirements.txt --no-cache-dir

# 소스 코드 복사 (설치 후에 복사하는 것이 도커 빌드 캐시 효율에 좋습니다)
COPY /app/ ./app/

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.12-slim AS runtime

WORKDIR /shopping_assistant

# 빌드된 가상환경과 소스코드 복사
COPY --from=builder /shopping_assistant/.venv /shopping_assistant/.venv
COPY --from=builder /shopping_assistant/app/ ./app/
COPY secrets/ ./secrets/

ENV PATH="/shopping_assistant/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SQL="/shopping_assistant/app/rag/sqllite/products.db"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
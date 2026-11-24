FROM python:3.11-slim

# 시스템 패키지 설치 (캐시됨)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gfortran build-essential python3-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# [핵심] 라이브러리 설치 (requirements.txt가 안 바뀌면 0.1초 만에 통과)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# [여기서부터만 새로 실행] 앱 코드 복사
COPY . .

RUN mkdir -p sim_workspace
EXPOSE 10000

CMD ["gunicorn", "--workers", "1", "--threads", "4", "--worker-class", "gthread", "--bind", "0.0.0.0:10000", "--timeout", "330", "--max-requests", "500", "--max-requests-jitter", "50", "--access-logfile", "-", "--error-logfile", "-", "flask_app:app"]
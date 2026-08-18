FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-speakers.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-speakers.txt \
    && pip install --no-cache-dir -r requirements-web.txt

COPY . .

ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn web_app:app --host ${HOST} --port ${PORT}"]

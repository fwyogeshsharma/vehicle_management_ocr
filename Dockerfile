FROM python:3.12-slim

# libgl1/libglib2.0-0: OpenCV. libgomp1: paddlepaddle's CPU runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PADDLE_PDX_MODEL_SOURCE=huggingface \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ocr ./ocr
COPY worker.py .

# Bake the models into the image so the container starts offline and a network blip
# can never turn into "every job failed". Same constructor arguments as ocr/engine.py.
RUN python -c "from ocr.engine import PaddleEngine; PaddleEngine()"

RUN useradd --create-home app \
    && mkdir -p /app/uploads \
    && cp -r /root/.paddlex /home/app/.paddlex \
    && chown -R app:app /home/app /app
USER app

CMD ["python", "worker.py"]

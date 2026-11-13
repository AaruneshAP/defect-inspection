# CPU inference image. Weights are NOT baked in: mount them and set MODEL_PATH (see docs/serving.md).
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/models/best.pt \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /srv

COPY requirements.txt constraints.txt ./

# CPU-only torch first (the +cpu build satisfies the pinned version), then the rest under the freeze.
# ultralytics pulls in GUI opencv-python; swap it for the headless build (no libGL needed).
RUN pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt -c constraints.txt \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install opencv-python-headless==5.0.0.93

COPY app ./app
COPY src ./src
COPY configs ./configs
COPY reports/phase3/frozen.json ./reports/phase3/frozen.json

RUN useradd --create-home --uid 10001 appuser && mkdir /models && chown appuser /models
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

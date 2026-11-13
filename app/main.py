"""FastAPI service: /health, /model-info, /predict. The model loads once at startup from MODEL_PATH."""

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from app.schemas import HealthResponse, ModelInfo, PredictResponse
from app.service import ALLOWED_CONTENT_TYPES, Detector, InspectionService, Settings, decode_image


def create_app(settings: Optional[Settings] = None, detector: Optional[Detector] = None,
               version: Optional[str] = None) -> FastAPI:
    """Build the app; tests inject a stub detector, production loads weights from settings.model_path."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = InspectionService(settings or Settings.from_env(), detector, version)
        yield

    app = FastAPI(title="Defect inspection API", lifespan=lifespan)

    def svc(request: Request) -> InspectionService:
        return request.app.state.service

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        """Liveness; reports whether weights are loaded."""
        return HealthResponse(model_loaded=svc(request).loaded)

    @app.get("/model-info", response_model=ModelInfo)
    def model_info(request: Request):
        """Model name/version and the decision policy in use."""
        return svc(request).info()

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: Request, files: List[UploadFile] = File(...), include_ignored: bool = False):
        """Detect defects and return a maintenance decision for each uploaded image."""
        s = svc(request)
        if not s.loaded:
            raise HTTPException(503, "model weights not loaded")
        if len(files) > s.settings.max_files:
            raise HTTPException(400, f"too many files (max {s.settings.max_files})")
        limit = s.settings.max_upload_bytes
        results = []
        for f in files:
            if f.content_type not in ALLOWED_CONTENT_TYPES:
                raise HTTPException(415, f"{f.filename}: content type {f.content_type!r} not allowed; "
                                         "use image/jpeg, image/png or image/bmp")
            data = f.file.read(limit + 1)
            if len(data) > limit:
                raise HTTPException(413, f"{f.filename}: larger than {limit} bytes")
            if not data:
                raise HTTPException(400, f"{f.filename}: empty file")
            try:
                img = decode_image(data)
            except ValueError as exc:
                raise HTTPException(400, f"{f.filename}: {exc}") from exc
            results.append(s.run(img, f.filename, include_ignored=include_ignored))
        return PredictResponse(model_version=s.version or "unknown", results=results)

    return app


app = create_app()

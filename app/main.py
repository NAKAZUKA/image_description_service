import uvicorn
from fastapi import FastAPI

from config import settings
from common.logger import setup_logger
from common.startup import startup_check
from api.v1.routes import image_ocr, inference_url, inference_file, streaming

def create_app() -> FastAPI:
    app = FastAPI(
        title="Qwen2.5-VL Service",
        description="Multimodal model for OCR, translations, image-based inference.",
        version="1.0.0"
    )

    setup_logger()

    @app.on_event("startup")
    def on_startup():
        startup_check()

    register_routes(app)

    return app

def register_routes(app: FastAPI):
    prefix = "/api/v1"
    app.include_router(image_ocr.router, prefix=prefix, tags=["image_ocr"])
    app.include_router(inference_url.router, prefix=prefix, tags=["inference_url"])
    app.include_router(inference_file.router, prefix=prefix, tags=["inference_file"])
    app.include_router(streaming.router, prefix=prefix, tags=["stream_inference"])


app = create_app()

if __name__ == '__main__':
    uvicorn.run('main:app', host=settings.HOST, port=settings.PORT, reload=False)

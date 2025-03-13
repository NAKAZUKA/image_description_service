import os
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from services.model_service import stream_inference

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/stream_inference_file")
async def streaming_file_endpoint(
    text: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        file_path = f"/uploads/{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)

        generator = stream_inference(
            prompt=text,
            local_path=file_path
        )

        response = StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )

        @response.on_shutdown
        def cleanup():
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    pass

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream_inference_url")
async def streaming_url_endpoint(
    text: str = Form(...),
    image_url: str = Form(...)
):
    try:
        logger.info(f"[STREAMING] Incoming request for streaming with prompt: {text}, image_url: {image_url}")

        generator = stream_inference(
            prompt=text,
            image_url=image_url
        )
        logger.info("[STREAMING] Streaming started successfully.")

        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )

    except Exception as e:
        logger.error(f"[STREAMING] Error during streaming: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

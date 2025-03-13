import os
import tempfile
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from services.model_service import run_inference, SYSTEM_PROMPT_OCR, check_model_ready
from services.parse_html_text import HTMLParser
from api.v1.schemas.request import ImageOCRRequest
from api.v1.schemas.response import OCRResponse
from common.errors import ModelLoadError

router = APIRouter()

@router.post("/image_ocr_url", response_model=OCRResponse)
def image_ocr(data: ImageOCRRequest):
    try:
        check_model_ready()

        raw_response = run_inference(
            image_url=data.image_url,
            prompt="QwenVL HTML",
            system_prompt=SYSTEM_PROMPT_OCR
        )

        parser = HTMLParser(html_text=raw_response, with_blocks=data.with_blocks)
        parsed_response = parser.parse()

        if data.with_blocks:
            return {"blocks": parsed_response["blocks"]}
        else:
            return {"text": parsed_response["text"]}

    except ModelLoadError as e:
        raise HTTPException(status_code=500, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/image_ocr_file", response_model=OCRResponse)
def image_ocr_file(
    file: UploadFile = File(...),
    with_blocks: bool = Form(False)
):
    tmp_path = None
    try:
        check_model_ready()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp_path = tmp.name
            tmp.write(file.file.read())

        raw_response = run_inference(local_path=tmp_path, prompt="QwenVL HTML", system_prompt=SYSTEM_PROMPT_OCR)

        parser = HTMLParser(html_text=raw_response, with_blocks=with_blocks)
        parsed_response = parser.parse()

        if with_blocks:
            return {"blocks": parsed_response["blocks"]}
        else:
            return {"text": parsed_response["text"]}

    except ModelLoadError as e:
        raise HTTPException(status_code=500, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

import os
import tempfile
from fastapi import APIRouter, HTTPException, File, Form, UploadFile

from services.model_service import run_inference
# from services.clean_response_for_inferense import clean_response
from services.translation_service import detect_language_local, translate_text
from api.v1.schemas.response import InferenceResponse
from common.errors import ModelLoadError, TranslationError

router = APIRouter()
SYSTEM_PROMPT_DEFAULT = "система \nТы умный ассистент для выполнения задач, основанных на образе. \nпользователь \nЧто это? \nПомощник \n"

@router.post("/inference_file", response_model=InferenceResponse)
def inference_file(
    text: str = Form(...),
    file: UploadFile = File(...)
):
    tmp_path = None
    try:
        lang_in = detect_language_local(text)
        text_en = text if lang_in in ["en", "und"] else translate_text(text, lang_in, "en")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp_path = tmp.name
            tmp.write(file.file.read())

        model_res_en = run_inference(local_path=tmp_path, prompt=text_en)

        lang_out = detect_language_local(model_res_en)
        translated_response = model_res_en if lang_out in ["ru", "und"] else translate_text(model_res_en, lang_out, "ru")
        # final_response = clean_response(translated_response, SYSTEM_PROMPT_DEFAULT)
        
        return InferenceResponse(content=translated_response)

    except ModelLoadError as e:
        raise HTTPException(status_code=500, detail=e.message)
    except TranslationError as e:
        raise HTTPException(status_code=500, detail=e.message)
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

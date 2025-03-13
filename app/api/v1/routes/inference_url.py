from fastapi import APIRouter, HTTPException

from services.model_service import run_inference
from services.translation_service import detect_language_local, translate_text
# from services.clean_response_for_inferense import clean_response
from api.v1.schemas.request import InferenceURLRequest
from api.v1.schemas.response import InferenceResponse
from common.errors import ModelLoadError, TranslationError

router = APIRouter()
SYSTEM_PROMPT_DEFAULT = "система \nТы умный ассистент для выполнения задач, основанных на образе. \nпользователь \nЧто это? \nПомощник \n"

@router.post("/inference_url", response_model=InferenceResponse)
def inference_url(data: InferenceURLRequest):
    try:
        lang_in = detect_language_local(data.text)
        text_en = data.text if lang_in in ["en", "und"] else translate_text(data.text, lang_in, "en")

        model_res_en = run_inference(image_url=data.image_url, prompt=text_en)

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

import requests
from langdetect import detect, DetectorFactory

from config import settings
from common.errors import TranslationError

DetectorFactory.seed = 0

def detect_language_local(text: str) -> str:
    try:
        return detect(text)
    except:
        return "und"

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    payload = {
        "text": text,
        "source_language": source_lang,
        "target_language": target_lang
    }
    try:
        r = requests.post(settings.TRANSLATOR_URL, json=payload, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return data.get("text", text)
        else:
            raise TranslationError(f"Translator service returned status {r.status_code}")
    except Exception as e:
        raise TranslationError(f"Translation failed: {str(e)}")

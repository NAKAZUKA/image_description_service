import re, logging, torch

from pathlib import Path
from threading import Thread
from typing import Generator
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, TextIteratorStreamer

from services.load_image import load_image
from config import settings
from common.errors import ModelLoadError


model = None
processor = None
device = None

SYSTEM_PROMPT_DEFAULT = "You are an intelligent Assistant for image-based tasks."
SYSTEM_PROMPT_OCR = "You are an AI specialized in recognizing and extracting text from images."

def load_model():
    global model, processor, device
    if model is not None and processor is not None:
        logging.info("[MODEL] Model already loaded, skipping initialization.")
        return

    model_path = Path(settings.MODEL_PATH)
    if not model_path.exists() or not any(model_path.iterdir()):
        raise ModelLoadError(f"[MODEL] Model directory is empty or missing: {model_path}")

    try:
        device = torch.device(settings.DEVICE)
        logging.info(f"[MODEL] Loading model from {model_path} to {device}...")

        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        processor_local = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model_local = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=quant_config
        )
        model_local.to(device)

        if model_local is None or processor_local is None:
            raise ModelLoadError("[MODEL] Model or processor was not properly initialized.")

        model, processor = model_local, processor_local
        logging.info("[MODEL] Model loaded successfully and ready for inference.")
    except Exception as e:
        logging.error(f"[MODEL] Failed to load model: {str(e)}")
        raise ModelLoadError(f"Unable to load the model: {str(e)}")

def check_model_ready():
    if model is None or processor is None:
        logging.error("[MODEL] Model is not loaded. Call `load_model()` first.")
        raise ModelLoadError("Model is not loaded. Please restart the service.")

def run_inference(image_url: str = None, local_path: str = None, prompt: str = "",
                  system_prompt: str = SYSTEM_PROMPT_DEFAULT) -> str:
    check_model_ready()
    image = load_image(image_url, local_path)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    final_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=final_prompt, images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=2048)

    result = processor.batch_decode(gen_ids, skip_special_tokens=True)
    raw_text = result[0] if isinstance(result, list) else result
    cleaned_text = re.sub(r'system.*?assistant', '', raw_text, flags=re.DOTALL).strip()
    return cleaned_text

def stream_inference(
    prompt: str,
    image_url: str = None,
    local_path: str = None,
    max_new_tokens: int = 1024,
    system_prompt: str = SYSTEM_PROMPT_DEFAULT
) -> Generator[str, None, None]:
    check_model_ready()
    logging.info("[STREAMING] Starting streaming inference...")

    try:
        image = load_image(image_url, local_path)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
        ]
        
        final_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=final_prompt, images=[image], return_tensors="pt").to(device)

        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_prompt=True,
            timeout=20.0,
            skip_special_tokens=True
        )

        generation_args = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id
        )

        thread = Thread(target=model.generate, kwargs=generation_args)
        thread.start()

        accumulated_text = ""
        for text_token in streamer:
            accumulated_text += text_token
            logging.info(f"[STREAMING] Generated token: {text_token}")
            yield accumulated_text + "\n"

        thread.join()

    except Exception as e:
        logging.error(f"[STREAMING] Streaming error: {str(e)}")
        yield f"Error: {str(e)}"
        raise

# app/common/startup.py
import os
import logging
import shutil
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

from config import settings
from common.errors import ModelLoadError
from services.model_service import load_model

def startup_check():
    logging.info("[STARTUP] Running startup check...")

    if not os.path.isdir(settings.MODEL_PATH) or not any(os.scandir(settings.MODEL_PATH)):
        logging.warning(f"[STARTUP] Model not found at {settings.MODEL_PATH}. Starting download...")
        download_model_if_needed()
    else:
        logging.info(f"[STARTUP] Model directory found: {settings.MODEL_PATH}")

    try:
        load_model()
        logging.info("[STARTUP] Model successfully initialized and ready to use.")
    except ModelLoadError as e:
        logging.critical(f"[STARTUP] Model failed to load: {str(e)}")
        raise SystemExit(1)

def download_model_if_needed():
    model_name = settings.MODEL_NAME
    model_path = settings.MODEL_PATH

    if os.path.exists(model_path) and any(os.scandir(model_path)):
        logging.info(f"[DOWNLOAD] Model already exists in {model_path}, skipping download.")
        return

    logging.info(f"[DOWNLOAD] Model not found, downloading from Hugging Face: {model_name}")

    try:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        _ = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        _ = AutoModelForImageTextToText.from_pretrained(model_name, quantization_config=quantization_config)

        logging.info("[DOWNLOAD] Model downloaded successfully.")
    except Exception as e:
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        raise ModelLoadError(f"[DOWNLOAD] Failed to download model: {str(e)}")


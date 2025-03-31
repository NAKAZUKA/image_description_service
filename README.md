# Qwen2.5-VL FastAPI Inference Service

## Описание проекта

Этот репозиторий представляет собой сервис на базе **FastAPI** для работы с мультимодальной моделью [`Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit`](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit) 🤗, способной обрабатывать изображения и текст.

Поддерживаются следующие функции:
- **OCR**: распознавание текста на изображении (по URL или файлу)
- **Inference**: описание содержимого изображения (по URL или файлу)
- **Streaming**: потоковый вывод результата описания изображения (chunked response)

---

## Структура проекта

```text
.
├── app
│   ├── api
│   │   └── v1
│   │       ├── routes                # FastAPI endpoints
│   │       └── schemas               # Pydantic-схемы запросов и ответов
│   ├── common                        # Общие утилиты и логгеры
│   ├── services                      # Логика обработки, работы с моделью и изображениями
│   ├── main.py                       # Точка входа FastAPI-приложения
│   └── config.py                     # Конфигурации
├── models_cache
│   └── Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit
├── poetry.lock
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Запуск через Docker

### Dockerfile

Используется базовый образ от Bitnami с оптимизированной сборкой PyTorch:

```dockerfile
FROM bitnami/pytorch:2.5.1 as production

USER root
ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app

RUN apt update && apt install -y \
    curl \
    build-essential \
    gcc \
    g++ \
    clang \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./poetry.lock ./poetry.lock
COPY ./pyproject.toml ./pyproject.toml

RUN python3 -m pip install poetry 
RUN poetry config virtualenvs.create false
RUN poetry config installer.max-workers 2
RUN poetry install --without dev --no-root

COPY . .

WORKDIR /app/app

ENTRYPOINT ["python3", "main.py"]
```

Образ адаптирован под `torch`, чтобы минимизировать занимаемое пространство, без потери производительности.

---

### docker-compose.yml

```yaml
version: '3.8'

services:
  qwen-vl_description:
    image: qwen-vl_description:latest
    restart: unless-stopped
    ports:
      - "8015:8015"
    env_file:
      - .env
    volumes:
      - './models_cache:/app/models_cache'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: [ "2" ]
              capabilities: [gpu]
```

- **Порт**: 8015
- **GPU**: указано использование конкретного GPU (device_ids: [ "2" ])
- **Volume**: модель монтируется из `./models_cache`
- **Переменные окружения**: загружаются из `.env`

---

## Модель

Используется модель:

> [Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit) 🤗

- Занимает ~10–11 GB видеопамяти на GPU
- Образ модели весит ~15 GB
- Размер на диске (в `models_cache`) — ~10 GB

---

## API

### 1. OCR

Распознавание текста на изображении:
- `POST /v1/ocr_url` — по URL изображения
- `POST /v1/ocr_file` — по загружаемому файлу

### 2. Inference

Описание объектов на изображении:
- `POST /v1/inference_url` — по URL изображения
- `POST /v1/inference_file` — по загружаемому файлу

### 3. Streaming

Потоковое описание изображения:
- `POST /v1/streaming_url` — по URL (с chunked-ответом)
- `POST /v1/streaming_file` — по файлу (с chunked-ответом)

---

## Установка зависимостей

Зависимости устанавливаются через [Poetry](https://python-poetry.org/):

```bash
poetry install --without dev --no-root
```

Для запуска вне Docker:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8015
```

---

## Примечания

- Для запуска требуется **NVIDIA GPU** с >=12 GB памяти
- Все веса модели загружаются заранее в `models_cache`
- Для потоковой генерации используется `StreamingResponse`

---

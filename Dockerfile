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

# RUN apt update 
# RUN apt install curl -y

COPY . .

WORKDIR /app/app

ENTRYPOINT ["python3", "main.py"]

FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04 as base

RUN apt update && \
apt install curl ca-certificates build-essential vim jq git -y && \
apt install --only-upgrade ca-certificates -y && \
apt-get clean && rm -rf /var/lib/apt/lists/

### Install python 3.13 with uv into a virtualenv (with pip) that is first on PATH
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv --seed --python ">=3.13.18,<3.14" /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

FROM base as build

WORKDIR /tmp
RUN pip install poetry poetry-plugin-export

COPY ./pyproject.toml ./poetry.lock* /tmp/

RUN poetry export --with dev --extras "vllm" -f requirements.txt --output requirements.txt --without-hashes

FROM base as runtime

WORKDIR /app
ENV PYTHONNOUSERSITE=1

COPY --from=build /tmp/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt
RUN python -m unidic download  # required for fugashi

# アプリケーションのコードをコピー

COPY docs /app/docs
COPY flexeval /app/flexeval
COPY tests /app/tests
COPY README.md /app/README.md
COPY ./pyproject.toml /app/
# git hash の情報をプログラム内で使用するためにコピー
COPY .git /app/.git

RUN pip install --no-cache-dir -e .

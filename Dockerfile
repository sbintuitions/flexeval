FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04 as base

### Install python 3.10 and set it as default python interpreter
RUN  apt update &&  apt install software-properties-common -y && \
add-apt-repository ppa:deadsnakes/ppa -y &&  apt update && \
apt install curl python3.10 build-essential vim git -y && \
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
apt install python3.10-venv python3.10-dev -y && \
curl -Ss https://bootstrap.pypa.io/get-pip.py | python3.10 && \
apt-get clean && rm -rf /var/lib/apt/lists/

FROM base as build

WORKDIR /tmp

RUN pip install poetry

COPY ./pyproject.toml ./poetry.lock* /tmp/

RUN poetry export --with dev --extras "vllm" -f requirements.txt --output requirements.txt --without-hashes

FROM base as runtime

WORKDIR /app

COPY --from=build /tmp/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt
RUN python -m unidic download  # required for fugashi

# アプリケーションのコードをコピー

COPY docs /app/docs
COPY flexeval /app/flexeval
COPY tests /app/tests
COPY internal /app/internal
COPY README.md /app/README.md
COPY ./pyproject.toml /app/
# git hash の情報をプログラム内で使用するためにコピー
COPY .git /app/.git

RUN pip install --no-cache-dir -e .
